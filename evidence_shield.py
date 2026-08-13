"""Evidence Shield: bounded ingestion and eight auditable research gates.

External material is data, never instructions.  The module extracts located
claims and records security/provenance decisions before any expert receives
content.  Raw bytes are deliberately absent from ``EvidencePacket``.
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Callable, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from PIL import Image
from pydantic import Field

from research_case import EvidenceDocument, FactClaim, GateDecision, StrictModel

PDF_MAX_BYTES = 20 * 1024 * 1024
IMAGE_MAX_BYTES = 10 * 1024 * 1024
URL_MAX_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 3

_PRIMARY_DOMAINS = {
    "federalreserve.gov", "bls.gov", "bea.gov", "treasury.gov", "cftc.gov",
    "imf.org", "worldbank.org", "gold.org",
}
_INJECTION_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in (
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"system\s+prompt",
        r"developer\s+message",
        r"do\s+not\s+follow\s+the\s+user",
        r"guaranteed\s+(buy|sell)\s+signal",
        r"\b(jailbreak|prompt\s+injection)\b",
        r"忽略.{0,8}(之前|上述).{0,8}指令",
        r"系统提示词",
    )
]
_DATE_RE = re.compile(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b")
_UNIT_RE = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>tonnes?|tons?|ounces?|oz|percent|%|bps?)\b",
    re.I,
)


class EvidenceShieldError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ValidatedURL(StrictModel):
    url: str
    host: str
    addresses: List[str]


class EvidencePacket(StrictModel):
    document: EvidenceDocument
    facts: List[FactClaim] = Field(default_factory=list)
    gates: List[GateDecision]
    confidence: float = Field(ge=0.0, le=1.0)
    safe_summary: str = ""

    @property
    def accepted_facts(self) -> List[FactClaim]:
        return [fact for fact in self.facts if fact.status == "accepted"]


def local_tesseract_ocr(payload: bytes) -> str:
    """Local-only OCR adapter used by the gateway when Tesseract is present."""
    import pytesseract

    with Image.open(BytesIO(payload)) as image:
        languages = set(pytesseract.get_languages(config=""))
        preferred = [language for language in ("chi_sim", "eng") if language in languages]
        language = "+".join(preferred) if preferred else None
        return pytesseract.image_to_string(image.convert("RGB"), lang=language, config="--psm 6")


def _is_public_address(raw: str) -> bool:
    address = ipaddress.ip_address(raw)
    return bool(address.is_global and not any((
        address.is_private, address.is_loopback, address.is_link_local,
        address.is_multicast, address.is_reserved, address.is_unspecified,
    )))


def validate_public_https_url(
    url: str,
    *,
    resolver: Callable[..., Any] = socket.getaddrinfo,
) -> ValidatedURL:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise EvidenceShieldError(
            "unsafe_url", "URL must resolve only to public HTTPS destinations"
        )
    host = parsed.hostname.rstrip(".").lower()
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        try:
            records = resolver(host, parsed.port or 443, type=socket.SOCK_STREAM)
        except (OSError, KeyError) as exc:
            raise EvidenceShieldError("dns_failed", "URL hostname could not be resolved") from exc
        addresses = sorted({str(record[4][0]) for record in records})
    if not addresses or any(not _is_public_address(address) for address in addresses):
        raise EvidenceShieldError(
            "unsafe_url", "URL must resolve only to public HTTPS destinations"
        )
    return ValidatedURL(url=url, host=host, addresses=addresses)


def _source_tier(host: Optional[str]) -> str:
    if not host:
        return "unknown"
    if any(host == domain or host.endswith("." + domain) for domain in _PRIMARY_DOMAINS):
        return "primary"
    return "secondary"


def _published_at(text: str) -> Optional[datetime]:
    match = _DATE_RE.search(text)
    if not match:
        return None
    try:
        return datetime.fromisoformat(match.group(1).replace("/", "-")).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _split_claims(text: str, locations: List[str], document_id: str) -> List[FactClaim]:
    claims: List[FactClaim] = []
    published = _published_at(text)
    for location in locations:
        if "\0" in location:
            locator, located_text = location.split("\0", 1)
        else:
            locator, located_text = location, text
        segments = re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", located_text)
        for segment in segments:
            cleaned = re.sub(r"\s+", " ", segment).strip()
            if len(cleaned) < 8:
                continue
            claim_id = "fact_" + hashlib.sha256(
                f"{document_id}|{locator}|{cleaned}".encode("utf-8")
            ).hexdigest()[:16]
            claims.append(FactClaim(
                claim_id=claim_id,
                document_id=document_id,
                text=cleaned[:4000],
                locator=locator,
                status="accepted",
                confidence=0.9,
                published_at=published,
            ))
            if len(claims) >= 60:
                return claims
    return claims


def _pdf_text(payload: bytes) -> tuple[str, List[str]]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(payload), strict=False)
        pages: List[str] = []
        combined: List[str] = []
        for index, page in enumerate(reader.pages[:100], start=1):
            text = (page.extract_text() or "").strip()
            if text:
                combined.append(text)
                pages.append(f"page {index}\0{text}")
        return "\n".join(combined), pages
    except Exception as exc:
        raise EvidenceShieldError("pdf_unparseable", "PDF could not be safely parsed", status_code=422) from exc


def _image_verified(payload: bytes) -> None:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
    except Exception as exc:
        raise EvidenceShieldError("image_signature_mismatch", "image signature does not match MIME") from exc


async def _fetch_public_url(
    url: str,
    *,
    client: httpx.AsyncClient,
    resolver: Callable[..., Any],
) -> tuple[str, bytes, str, str]:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        validated = validate_public_https_url(current, resolver=resolver)
        response = await client.get(validated.url, follow_redirects=False)
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            if not location:
                raise EvidenceShieldError("redirect_missing", "redirect response has no location")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        content = response.content
        if len(content) > URL_MAX_BYTES:
            raise EvidenceShieldError("url_too_large", "URL response exceeds 5 MB", status_code=413)
        content_type = response.headers.get("content-type", "text/html").split(";", 1)[0].lower()
        return current, content, content_type, validated.host
    raise EvidenceShieldError("too_many_redirects", "URL exceeded redirect limit")


def _gate(
    gate: str,
    decision: str,
    reason: str,
    multiplier: float,
    refs: Optional[List[str]] = None,
) -> GateDecision:
    return GateDecision(
        gate=gate,
        decision=decision,
        reason=reason,
        confidence_multiplier=multiplier,
        evidence_refs=refs or [],
    )


def _detect_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def _deduped(text: str) -> bool:
    parts = [re.sub(r"\W+", " ", part.lower()).strip() for part in re.split(r"[.!?。！？]", text)]
    meaningful = [part for part in parts if len(part) >= 12]
    return len(meaningful) != len(set(meaningful))


def _unit_conflict(text: str) -> bool:
    by_number: dict[str, set[str]] = {}
    for match in _UNIT_RE.finditer(text):
        number = match.group("number")
        unit = match.group("unit").lower()
        by_number.setdefault(number, set()).add(unit)
    incompatible = ({"tonne", "tonnes", "ton", "tons"}, {"ounce", "ounces", "oz"})
    for units in by_number.values():
        if any(units & left and units & right for left in incompatible for right in incompatible if left is not right):
            return True
    return False


async def ingest_evidence(
    *,
    question: str,
    url: Optional[str] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    payload: Optional[bytes] = None,
    http: Optional[httpx.AsyncClient] = None,
    resolver: Callable[..., Any] = socket.getaddrinfo,
    ocr: Optional[Callable[[bytes], str]] = None,
    now: Optional[datetime] = None,
) -> EvidencePacket:
    if url and payload is not None:
        raise EvidenceShieldError("multiple_inputs", "provide one primary evidence input")
    now = now or datetime.now(timezone.utc)
    own_http = http is None
    client = http or httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=3.0))
    host: Optional[str] = None
    source_url: Optional[str] = None
    kind = "question"
    text = ""
    locations: List[str] = []
    extraction_status = "complete"
    degradation: List[str] = []
    raw = payload or b""

    try:
        if url:
            source_url, raw, content_type, host = await _fetch_public_url(
                url, client=client, resolver=resolver
            )
            kind = "url"
            if content_type == "application/pdf":
                if not raw.startswith(b"%PDF"):
                    raise EvidenceShieldError("mime_signature_mismatch", "PDF signature does not match MIME")
                text, locations = _pdf_text(raw)
            else:
                decoded = raw.decode("utf-8", errors="replace")
                text = BeautifulSoup(decoded, "html.parser").get_text(" ", strip=True)
                locations = [source_url]
        elif payload is not None:
            mime = (content_type or "").lower().split(";", 1)[0]
            if mime == "application/pdf":
                kind = "pdf"
                if len(payload) > PDF_MAX_BYTES:
                    raise EvidenceShieldError("pdf_too_large", "PDF exceeds 20 MB", status_code=413)
                if not payload.startswith(b"%PDF"):
                    raise EvidenceShieldError("mime_signature_mismatch", "PDF signature does not match MIME")
                text, locations = _pdf_text(payload)
            elif mime in {"image/png", "image/jpeg", "image/webp"}:
                kind = "image"
                if len(payload) > IMAGE_MAX_BYTES:
                    raise EvidenceShieldError("image_too_large", "image exceeds 10 MB", status_code=413)
                _image_verified(payload)
                if ocr is None:
                    extraction_status = "abstain"
                    degradation.append("local_ocr_unavailable_or_complex_chart")
                else:
                    try:
                        text = (ocr(payload) or "").strip()
                    except Exception:
                        text = ""
                        degradation.append("local_ocr_failed")
                    if text:
                        locations = ["image full-frame"]
                    else:
                        extraction_status = "abstain"
                        if "local_ocr_failed" not in degradation:
                            degradation.append("ocr_returned_no_reliable_text")
            else:
                raise EvidenceShieldError("unsupported_mime", "only PDF, PNG, JPEG and WebP are accepted", status_code=415)
        else:
            raw = question.encode("utf-8")
            content_type = "text/plain"
            extraction_status = "abstain"
            degradation.append("no_external_evidence")

        sha = hashlib.sha256(raw).hexdigest()
        document_id = "doc_" + sha[:16]
        facts = _split_claims(text, locations, document_id) if text and locations else []
        fact_ids = [fact.claim_id for fact in facts]
        published = _published_at(text)
        attack = _detect_injection(text)
        duplicate = _deduped(text)
        unit_conflict = _unit_conflict(text)
        stale = bool(published and (now - published).days > 180)

        gates = [
            _gate("access", "pass", "bounded input and signature checks passed", 1.0),
            _gate(
                "provenance_time", "review" if stale else "pass",
                "published material is older than 180 days" if stale else "source and time metadata recorded",
                0.7 if stale else 1.0, fact_ids,
            ),
            _gate(
                "fact_location", "pass" if facts else "abstain",
                "claims have page or image/url locators" if facts else "no reliably located text extracted",
                1.0 if facts else 0.0, fact_ids,
            ),
            _gate(
                "ai_attack", "block" if attack else "pass",
                "instruction-like carrier detected and isolated" if attack else "no instruction carrier detected",
                0.0 if attack else 1.0, fact_ids,
            ),
            _gate(
                "dedup_replay", "review" if duplicate else "pass",
                "repeated claims require replay review" if duplicate else "no repeated claim cluster detected",
                0.8 if duplicate else 1.0, fact_ids,
            ),
            _gate(
                "consistency", "review" if unit_conflict else "pass",
                "same value appears with incompatible units" if unit_conflict else "no deterministic unit conflict detected",
                0.6 if unit_conflict else 1.0, fact_ids,
            ),
            _gate(
                "market_coherence", "review" if facts else "abstain",
                "downstream orchestrator must compare evidence with market response" if facts else "no facts to compare",
                0.9 if facts else 0.0, fact_ids,
            ),
            _gate(
                "output_audit", "block" if attack else ("pass" if facts else "abstain"),
                "unsafe carrier blocks downstream output" if attack else (
                    "accepted claims are traceable" if facts else "no evidence-backed output available"
                ),
                0.0 if attack or not facts else 1.0, fact_ids,
            ),
        ]
        if attack:
            facts = [fact.model_copy(update={"status": "blocked", "confidence": 0.0}) for fact in facts]
        multiplier = 1.0
        for gate in gates:
            multiplier *= gate.confidence_multiplier
        document = EvidenceDocument(
            document_id=document_id,
            kind=kind,
            filename=filename,
            sha256=sha,
            source_url=source_url,
            content_type=content_type or "application/octet-stream",
            source_tier=_source_tier(host),
            retrieved_at=now,
            published_at=published,
            persisted_raw=False,
            extraction_status=extraction_status,
            degradation_flags=degradation,
        )
        safe_summary = " ".join(fact.text for fact in facts if fact.status == "accepted")[:1200]
        return EvidencePacket(
            document=document,
            facts=facts,
            gates=gates,
            confidence=round(multiplier, 6),
            safe_summary=safe_summary,
        )
    finally:
        if own_http:
            await client.aclose()
