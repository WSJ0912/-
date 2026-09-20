const assert = require("node:assert/strict");
const test = require("node:test");

const { rendererRouteAllowed } = require("../electron/ipc-policy.cjs");

test("renderer allows only the expected method and controlled identifier routes", () => {
  assert.equal(rendererRouteAllowed("/api/studies", "GET"), true);
  assert.equal(rendererRouteAllowed("/api/studies/ST-A1B2/predict", "POST"), true);
  assert.equal(rendererRouteAllowed("/api/reports", "GET"), true);
  assert.equal(rendererRouteAllowed("/api/reports/RPT-A1B2/revisions", "GET"), true);
  assert.equal(rendererRouteAllowed("/api/predictions/PRD-A1B2/cams/0", "GET"), true);
  assert.equal(rendererRouteAllowed("/api/predictions/PRD-A1B2/cams/13", "GET"), true);
  assert.equal(rendererRouteAllowed("/api/studies/ST-A1B2/predict", "GET"), false);
  assert.equal(rendererRouteAllowed("/api/studies/../../secret/preview", "GET"), false);
  assert.equal(rendererRouteAllowed("/api/reports/../../secret/revisions", "GET"), false);
  assert.equal(rendererRouteAllowed("/api/predictions/PRD-A1B2/cams/14", "GET"), false);
  assert.equal(rendererRouteAllowed("/api/predictions/PRD-A1B2/cams/-1", "GET"), false);
  assert.equal(rendererRouteAllowed("/api/predictions/PRD-A1B2/cams/0", "POST"), false);
});

test("renderer cannot submit local paths through privileged service routes", () => {
  assert.equal(rendererRouteAllowed("/api/import/stage", "POST"), false);
  assert.equal(rendererRouteAllowed("/api/models/install", "POST"), false);
  assert.equal(rendererRouteAllowed("/api/experiments/import", "POST"), false);
  assert.equal(rendererRouteAllowed("/api/reports/export", "POST"), false);
  assert.equal(rendererRouteAllowed("/api/logout", "POST"), false);
});
