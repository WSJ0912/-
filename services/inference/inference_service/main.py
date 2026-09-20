from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .assistant import OpenAIAssistant
from .contracts import (
    AssistantExperimentRequest,
    AssistantReportRequest,
    CommitImportRequest,
    ConfirmReportRequest,
    CreateUserRequest,
    ExportReportRequest,
    InstallModelRequest,
    LoginRequest,
    ReportDraftRequest,
    ReviewRequest,
    SetupRequest,
    StageImportRequest,
)
from .core import PlatformCore
from .security import LocalTokenStore, Role, SessionStore, UserSession, require_permission


def create_app(root: str | Path = "./runtime") -> Any:
    """Build the API; imports FastAPI only when the service is actually run."""

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("fastapi is required to run the local inference service") from exc

    app = FastAPI(title="Medical Imaging Local Service", version="0.1.0", docs_url=None, redoc_url=None)
    core = PlatformCore(root)
    process_tokens = LocalTokenStore()
    sessions = SessionStore()
    app.state.core = core
    app.state.process_token = process_tokens.token

    def process_auth(authorization: str | None = Header(default=None)) -> None:
        candidate = authorization.removeprefix("Bearer ").strip() if authorization else None
        if not process_tokens.validate(candidate):
            raise HTTPException(status_code=401, detail="invalid local service token")

    def session_auth(
        authorization: str | None = Header(default=None),
        x_session_token: str | None = Header(default=None),
    ) -> UserSession:
        # The process token authenticates the local transport; the separate
        # session token authenticates the logged-in role.
        process_candidate = authorization.removeprefix("Bearer ").strip() if authorization else None
        if not process_tokens.validate(process_candidate):
            raise HTTPException(status_code=401, detail="invalid local service token")
        session = sessions.get(x_session_token)
        if session is None:
            raise HTTPException(status_code=401, detail="login required")
        return session

    def permission(permission_name: str) -> Callable[[UserSession], UserSession]:
        def dependency(session: UserSession = Depends(session_auth)) -> UserSession:
            try:
                require_permission(session.role, permission_name)
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            return session

        return dependency

    def operation(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    def health(_: None = Depends(process_auth)) -> dict[str, Any]:
        return {"ok": True, "service": "local-inference", "host": "127.0.0.1", "model": core.models.status()}

    @app.get("/api/status")
    def status(_: None = Depends(process_auth)) -> dict[str, Any]:
        return {**core.setup_status(), "model": core.models.status()}

    @app.post("/api/setup")
    def setup(request: SetupRequest, _: None = Depends(process_auth)) -> dict[str, str]:
        def run() -> dict[str, str]:
            user_id = core.create_initial_admin(request.username, request.password)
            return {"userId": user_id, "role": Role.ADMIN.value, "sessionToken": sessions.create(user_id, Role.ADMIN)}

        return operation(run)

    @app.post("/api/login")
    def login(request: LoginRequest, _: None = Depends(process_auth)) -> dict[str, str]:
        def run() -> dict[str, str]:
            session = core.authenticate(request.username, request.password)
            return {"sessionToken": sessions.create(session.user_id, session.role), "role": session.role.value, "userId": session.user_id}

        return operation(run)

    @app.post("/api/logout")
    def logout(
        session: UserSession = Depends(session_auth),
        x_session_token: str | None = Header(default=None),
    ) -> dict[str, bool]:
        del session
        sessions.revoke(x_session_token)
        return {"loggedOut": True}

    @app.post("/api/users/doctors")
    def create_doctor(request: CreateUserRequest, session: UserSession = Depends(permission("manage_users"))) -> dict[str, str]:
        return operation(lambda: {"userId": core.create_doctor(session, request.username, request.password), "role": Role.DOCTOR.value})

    @app.get("/api/users")
    def users(session: UserSession = Depends(permission("manage_users"))) -> list[dict[str, Any]]:
        return operation(core.database.list_users)

    @app.post("/api/import/stage")
    def stage(request: StageImportRequest, session: UserSession = Depends(permission("import"))) -> list[dict[str, Any]]:
        return operation(lambda: core.stage_imports(session, request.paths))

    @app.post("/api/import/{staging_id}/commit")
    def commit(staging_id: str, request: CommitImportRequest, session: UserSession = Depends(permission("import"))) -> dict[str, Any]:
        return operation(
            lambda: core.commit_import(
                session,
                staging_id,
                adult_confirmed=request.adultConfirmed,
                chest_confirmed=request.chestConfirmed,
                view_confirmed=request.viewConfirmed,
                view_position=request.viewPosition,
                burned_in_reviewed=request.burnedInReviewed,
                rectangles=[item.model_dump() for item in request.rectangles],
            )
        )

    @app.delete("/api/import/{staging_id}")
    def cancel(staging_id: str, session: UserSession = Depends(permission("import"))) -> dict[str, bool]:
        operation(lambda: core.cancel_import(session, staging_id))
        return {"deleted": True}

    @app.get("/api/studies")
    def studies(session: UserSession = Depends(permission("import"))) -> list[dict[str, Any]]:
        return operation(lambda: core.list_studies(session))

    @app.get("/api/import/{staging_id}/preview")
    def staged_preview(staging_id: str, session: UserSession = Depends(permission("import"))) -> dict[str, Any]:
        return operation(lambda: core.preview_staged(session, staging_id))

    @app.get("/api/studies/{study_id}/preview")
    def study_preview(study_id: str, session: UserSession = Depends(permission("review"))) -> dict[str, Any]:
        return operation(lambda: core.preview_study(session, study_id))

    @app.post("/api/models/install")
    def install_model(request: InstallModelRequest, session: UserSession = Depends(permission("manage_models"))) -> dict[str, Any]:
        return operation(lambda: core.install_model(session, request.packagePath))

    @app.get("/api/models/active")
    def active_model(_: UserSession = Depends(session_auth)) -> dict[str, Any]:
        return core.models.status()

    @app.post("/api/studies/{study_id}/predict")
    def predict(study_id: str, session: UserSession = Depends(permission("infer"))) -> dict[str, Any]:
        return operation(lambda: core.infer(session, study_id))

    @app.get("/api/predictions/{prediction_id}/cams/{label_index}")
    def prediction_cam(
        prediction_id: str,
        label_index: int,
        session: UserSession = Depends(permission("review")),
    ) -> dict[str, Any]:
        return operation(
            lambda: core.prediction_cam(session, prediction_id, label_index)
        )

    @app.post("/api/reviews")
    def review(request: ReviewRequest, session: UserSession = Depends(permission("review"))) -> dict[str, Any]:
        return operation(lambda: core.save_review(session, request.studyId, request.predictionId, request.decisions, request.notes))

    @app.post("/api/reports/draft")
    def report_draft(request: ReportDraftRequest, session: UserSession = Depends(permission("report"))) -> dict[str, Any]:
        return operation(lambda: core.save_report_draft(session, request.studyId, request.body, request.reportId, request.reviewId))

    @app.get("/api/reports")
    def reports(
        session: UserSession = Depends(permission("report")),
    ) -> list[dict[str, Any]]:
        return operation(lambda: core.list_reports(session))

    @app.get("/api/reports/{report_id}/revisions")
    def report_revisions(
        report_id: str,
        session: UserSession = Depends(permission("report")),
    ) -> list[dict[str, Any]]:
        return operation(lambda: core.report_history(session, report_id))

    @app.post("/api/reports/confirm")
    def report_confirm(request: ConfirmReportRequest, session: UserSession = Depends(permission("confirm_report"))) -> dict[str, Any]:
        return operation(lambda: core.confirm_report(session, request.reportId, request.revision))

    @app.post("/api/reports/export")
    def report_export(request: ExportReportRequest, session: UserSession = Depends(permission("report"))) -> dict[str, str]:
        return operation(lambda: {"path": core.export_report(session, request.reportId, request.revision, request.outputPath)})

    @app.post("/api/experiments/import")
    def experiment_import(request: InstallModelRequest, session: UserSession = Depends(permission("experiment"))) -> dict[str, Any]:
        return operation(lambda: core.import_experiment(session, request.packagePath))

    @app.post("/api/assistant/report")
    def assistant_report(
        request: AssistantReportRequest,
        session: UserSession = Depends(permission("assistant")),
        x_assistant_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        assistant = OpenAIAssistant(api_key=x_assistant_key)

        def run() -> dict[str, Any]:
            try:
                if request.studyId is None:
                    if assistant.available:
                        raise ValueError("studyId is required for assistant context")
                    # Preserve the old no-key offline workflow without trusting
                    # any client-supplied observations or review values.
                    return assistant.report_draft(
                        {
                            "observations": [],
                            "review": {"decisions": {}, "notes": ""},
                            "clinicianText": request.clinicianText,
                        }
                    )
                return core.assistant_report(
                    session,
                    request.studyId,
                    request.clinicianText,
                    request.reviewId,
                    assistant,
                )
            except RuntimeError as exc:
                # Offline mode remains useful without pretending an LLM ran.
                return {
                    "draft": request.clinicianText,
                    "cautions": [
                        str(exc),
                        "本结果为离线草稿，必须由医生编辑并确认",
                    ],
                }

        return operation(run)

    @app.post("/api/assistant/experiment")
    def assistant_experiment(
        request: AssistantExperimentRequest,
        session: UserSession = Depends(permission("assistant")),
        x_assistant_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        try:
            return OpenAIAssistant(api_key=x_assistant_key).experiment_summary(payload)
        except RuntimeError as exc:
            return {"summary": request.experimentNotes, "limitations": [str(exc), "未调用在线实验助手"]}

    @app.exception_handler(Exception)
    async def unhandled(_: Request, error: Exception) -> Any:
        # Do not leak local paths, stack traces or database details to the renderer.
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=500, content={"detail": "internal service error"})

    return app
