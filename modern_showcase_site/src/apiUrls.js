export function deriveApiEndpoint(analyzeUrl, endpointPath) {
  const normalizedPath = endpointPath.startsWith('/') ? endpointPath : `/${endpointPath}`;

  try {
    const base = new URL(analyzeUrl);
    return new URL(normalizedPath, base.origin).toString();
  } catch {
    return normalizedPath;
  }
}
