const RENDERER_ROUTES = Object.freeze([
  ["GET", /^\/health$/],
  ["GET", /^\/api\/status$/],
  ["POST", /^\/api\/(setup|login)$/],
  ["GET", /^\/api\/users$/],
  ["POST", /^\/api\/users\/doctors$/],
  ["GET", /^\/api\/import\/TMP-[A-Z0-9-]+\/preview$/],
  ["POST", /^\/api\/import\/TMP-[A-Z0-9-]+\/commit$/],
  ["DELETE", /^\/api\/import\/TMP-[A-Z0-9-]+$/],
  ["GET", /^\/api\/studies$/],
  ["GET", /^\/api\/studies\/ST-[A-Z0-9-]+\/preview$/],
  ["POST", /^\/api\/studies\/ST-[A-Z0-9-]+\/predict$/],
  ["GET", /^\/api\/predictions\/PRD-[A-Z0-9-]+\/cams\/(?:[0-9]|1[0-3])$/],
  ["GET", /^\/api\/models\/active$/],
  ["POST", /^\/api\/reviews$/],
  ["GET", /^\/api\/reports$/],
  ["GET", /^\/api\/reports\/RPT-[A-Z0-9-]+\/revisions$/],
  ["POST", /^\/api\/reports\/(draft|confirm)$/],
  ["POST", /^\/api\/assistant\/(report|experiment)$/],
]);

function rendererRouteAllowed(endpoint, method) {
  if (typeof endpoint !== "string" || typeof method !== "string") return false;
  const normalizedMethod = method.toUpperCase();
  return RENDERER_ROUTES.some(
    ([allowedMethod, pattern]) => allowedMethod === normalizedMethod && pattern.test(endpoint),
  );
}

module.exports = { rendererRouteAllowed };
