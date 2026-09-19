from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from xml.sax.saxutils import escape


def export_report_pdf(
    output_path: str | Path,
    report: Mapping[str, Any],
    prediction: Mapping[str, Any],
    review: Mapping[str, Any] | None = None,
) -> Path:
    """Export clinician text followed by an immutable AI provenance appendix."""

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("reportlab is required for PDF export") from exc
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    for style_name in ("Title", "Normal", "BodyText", "Heading2"):
        styles[style_name].fontName = "STSong-Light"
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm)
    story = [
        Paragraph("胸部 X 光阅片报告", styles["Title"]),
        Paragraph(f"检查编号：{escape(str(report['studyId']))} | 版本：{report['revision']} | 状态：{escape(str(report['status']))}", styles["Normal"]),
        Spacer(1, 8),
        Paragraph(escape(str(report["body"])).replace("\n", "<br/>"), styles["BodyText"]),
        Spacer(1, 16),
        Paragraph("AI 附录（不可作为独立诊断）", styles["Heading2"]),
        Paragraph(
            f"模型：{escape(str(prediction['modelId']))} {escape(str(prediction['modelVersion']))} | "
            f"来源：{escape(str(prediction['source']))}<br/>"
            f"模型 SHA-256：{escape(str(prediction.get('modelSha256') or '--'))}<br/>"
            f"清单 SHA-256：{escape(str(prediction.get('manifestSha256') or '--'))}<br/>"
            f"预测记录：{escape(str(prediction['predictionId']))}",
            styles["Normal"],
        ),
    ]
    decisions = (review or {}).get("decisions", {})
    if review:
        story.extend(
            [
                Spacer(1, 6),
                Paragraph(
                    f"复核记录：{escape(str(review['reviewId']))} | 医生账号："
                    f"{escape(str(review['doctorId']))} | 时间：{escape(str(review['createdAt']))}<br/>"
                    f"备注：{escape(str(review.get('notes') or '--'))}",
                    styles["Normal"],
                ),
            ]
        )
    rows = [["观察项", "原始概率", "阈值", "医生复核"]]
    for label, probability in prediction["probabilities"].items():
        probability_text = "--" if probability is None else f"{float(probability):.4f}"
        threshold = prediction.get("thresholds", {}).get(label)
        rows.append([label, probability_text, "--" if threshold is None else f"{float(threshold):.4f}", decisions.get(label, "--")])
    table = Table(rows, repeatRows=1, colWidths=[64 * mm, 30 * mm, 30 * mm, 36 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7edf0")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#86969d")), ("FONT", (0, 0), (-1, -1), "STSong-Light", 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(Spacer(1, 8))
    story.append(table)
    doc.build(story)
    return destination
