// Tiny pub/sub search index. Pages register entries when their data loads;
// the GlobalSearch overlay subscribes and queries the merged index.
// Entry shape: { id, source, title, hint, route, hash?, keywords? }

const entries = new Map(); // source -> entry[]
const listeners = new Set();

export function addSearchEntries(source, items) {
  entries.set(
    source,
    (items || []).filter((item) => item && item.title),
  );
  listeners.forEach((listener) => listener());
}

export function clearSearchEntries(source) {
  entries.delete(source);
  listeners.forEach((listener) => listener());
}

export function subscribeSearch(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function querySearch(rawQuery, limit = 12) {
  const query = (rawQuery || '').trim().toLowerCase();
  const all = [...entries.values()].flat();
  if (!query) return all.slice(0, limit);

  const terms = query.split(/\s+/).filter(Boolean);
  const scored = [];
  for (const entry of all) {
    const haystack = `${entry.title} ${entry.hint || ''} ${(entry.keywords || []).join(' ')}`
      .toLowerCase();
    let score = 0;
    for (const term of terms) {
      if (!haystack.includes(term)) {
        score = -1;
        break;
      }
      score += entry.title.toLowerCase().includes(term) ? 3 : 1;
    }
    if (score > 0) scored.push([score, entry]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, limit).map(([, entry]) => entry);
}
