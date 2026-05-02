from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


SOURCE_XLSX = Path("/Users/mansur/Downloads/dannye-ce1e2d0f-249c-4782-a612-f0d6eeba56c0.xlsx")
OUT_DIR = Path("/Users/mansur/Desktop/Course3Project/outputs")
PNG_PATH = OUT_DIR / "mobicar_funnel_report.png"
PDF_PATH = OUT_DIR / "mobicar_funnel_report.pdf"

W = 2200
H = 1400

FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"

BG = "#f6f3ee"
CARD = "#fffdf9"
INK = "#16181d"
TEXT = "#2b3441"
MUTED = "#667085"
GRID = "#ddd5c8"
LINE = "#e9e2d7"
ACCENT = "#f59e0b"
ACCENT_SOFT = "#ffefc7"
DARK = "#111827"
GREY = "#a7b0bd"
GREY_SOFT = "#d7dde5"
SUCCESS = "#0ea5a4"
SHADOW = (17, 24, 39, 26)


def f_reg(size: int):
    return ImageFont.truetype(FONT_REG, size=size)


def f_bold(size: int):
    return ImageFont.truetype(FONT_BOLD, size=size)


def f_black(size: int):
    return ImageFont.truetype(FONT_BLACK, size=size)


def rr(draw, box, fill, outline=None, width=1, radius=28):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def shadow_card(base: Image.Image, box, radius=30, blur=20, offset=(0, 10)):
    shadow = Image.new("RGBA", base.size, (255, 255, 255, 0))
    sdraw = ImageDraw.Draw(shadow)
    x1, y1, x2, y2 = box
    dx, dy = offset
    sdraw.rounded_rectangle((x1 + dx, y1 + dy, x2 + dx, y2 + dy), radius=radius, fill=SHADOW)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(base, shadow)


def add_blob(base: Image.Image, box, color, blur=42):
    layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(box, fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(base, layer)


def text_box(draw, text, font, max_width, spacing=7):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines), len(lines) * (font.size + spacing)


def multiline(draw, xy, text, font, fill, max_width, spacing=8):
    wrapped, _ = text_box(draw, text, font, max_width, spacing=spacing)
    draw.multiline_text(xy, wrapped, font=font, fill=fill, spacing=spacing)
    return wrapped


def pill(draw, x, y, w, label, value, accent_fill):
    rr(draw, (x, y, x + w, y + 110), CARD, outline=LINE, width=2, radius=24)
    rr(draw, (x + 22, y + 20, x + 88, y + 52), accent_fill, radius=16)
    draw.text((x + 106, y + 20), label, font=f_reg(21), fill=MUTED)
    draw.text((x + 22, y + 64), value, font=f_black(30), fill=INK)


def load_metrics():
    df = pd.read_excel(SOURCE_XLSX)
    df = df[df["Неделя"].astype(str).str.startswith("Неделя")].copy()
    df["week_num"] = df["Неделя"].str.extract(r"(\d+)").astype(int)
    weekly = df.groupby("week_num", as_index=False).sum(numeric_only=True).sort_values("week_num")

    weekly["inst_reg"] = weekly["Регистрации"] / weekly["Установки"]
    weekly["reg_search"] = weekly["Открыли поиск"] / weekly["Регистрации"]
    weekly["search_view"] = weekly["Просмотрели авто"] / weekly["Открыли поиск"]
    weekly["view_book"] = weekly["Забронировали"] / weekly["Просмотрели авто"]
    weekly["book_trip"] = weekly["Первая поездка"] / weekly["Забронировали"]
    weekly["overall"] = weekly["Первая поездка"] / weekly["Установки"]

    base = weekly[weekly["week_num"] <= 6]
    recent = weekly[weekly["week_num"] >= 7]

    return {
        "weekly": weekly,
        "overall_before": base["overall"].mean(),
        "overall_after": recent["overall"].mean(),
        "reg_search_before": base["reg_search"].mean(),
        "reg_search_after": recent["reg_search"].mean(),
        "installs_growth": recent["Установки"].mean() / base["Установки"].mean() - 1,
        "step_labels": [
            "Установка →\nрегистрация",
            "Регистрация →\nпоиск",
            "Поиск →\nпросмотр",
            "Просмотр →\nбронь",
            "Бронь →\nпоездка",
        ],
        "step_before": [
            base["inst_reg"].mean(),
            base["reg_search"].mean(),
            base["search_view"].mean(),
            base["view_book"].mean(),
            base["book_trip"].mean(),
        ],
        "step_after": [
            recent["inst_reg"].mean(),
            recent["reg_search"].mean(),
            recent["search_view"].mean(),
            recent["view_book"].mean(),
            recent["book_trip"].mean(),
        ],
    }


def draw_line_chart(draw, weekly, box):
    x1, y1, x2, y2 = box
    top_pad = 65
    left_pad = 86
    right_pad = 48
    bottom_pad = 72
    px1 = x1 + left_pad
    px2 = x2 - right_pad
    py1 = y1 + top_pad
    py2 = y2 - bottom_pad
    width = px2 - px1
    height = py2 - py1

    draw.text((x1 + 28, y1 + 18), "Динамика общей конверсии по неделям", font=f_bold(25), fill=INK)
    draw.text((x1 + 28, y1 + 50), "install → first trip", font=f_reg(19), fill=MUTED)

    y_min = 17
    y_max = 29
    weeks = weekly["week_num"].tolist()
    values = (weekly["overall"] * 100).tolist()

    def x_pos(idx):
        return px1 + idx * width / (len(weeks) - 1)

    def y_pos(val):
        return py2 - (val - y_min) / (y_max - y_min) * height

    shade_left = x_pos(5) + width / (len(weeks) - 1) * 0.5
    rr(draw, (shade_left, py1 - 10, px2 + 18, py2 + 8), ACCENT_SOFT, radius=24)

    for tick in [18, 21, 24, 27]:
        y = y_pos(tick)
        draw.line((px1, y, px2, y), fill=GRID, width=2)
        draw.text((px1 - 54, y - 12), f"{tick}%", font=f_reg(18), fill=MUTED)

    points = [(x_pos(i), y_pos(v)) for i, v in enumerate(values)]
    draw.line(points, fill=DARK, width=8, joint="curve")
    for x, y in points:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=DARK)

    for i, week in enumerate(weeks):
        draw.text((x_pos(i) - 8, py2 + 22), str(week), font=f_reg(19), fill=MUTED)

    draw.text((shade_left + 24, py1 - 40), "Недели 7–8", font=f_bold(18), fill=INK)
    draw.text((px2 - 30, y_pos(values[-1]) - 40), "19.1%", font=f_black(24), fill=DARK)

    note_box = (x1 + 245, y1 + 150, x1 + 610, y1 + 242)
    rr(draw, note_box, CARD, outline="#e9d39f", width=2, radius=22)
    draw.text((note_box[0] + 22, note_box[1] + 20), "Общая конверсия держалась на 26–27%", font=f_bold(21), fill=INK)
    draw.text((note_box[0] + 22, note_box[1] + 50), "и резко упала только после изменений", font=f_reg(21), fill=MUTED)


def draw_bar_chart(draw, labels, before, after, box):
    x1, y1, x2, y2 = box
    left_pad = 72
    right_pad = 36
    top_pad = 84
    bottom_pad = 82
    px1 = x1 + left_pad
    px2 = x2 - right_pad
    py1 = y1 + top_pad
    py2 = y2 - bottom_pad
    width = px2 - px1
    height = py2 - py1

    draw.text((x1 + 28, y1 + 20), "Где именно сломалась воронка", font=f_bold(25), fill=INK)
    draw.text((x1 + 28, y1 + 52), "Средняя конверсия шага: недели 1–6 vs недели 7–8", font=f_reg(19), fill=MUTED)

    y_min = 40
    y_max = 95

    def y_pos(val):
        return py2 - (val - y_min) / (y_max - y_min) * height

    for tick in [40, 50, 60, 70, 80, 90]:
        y = y_pos(tick)
        draw.line((px1, y, px2, y), fill=GRID, width=2)
        draw.text((px1 - 52, y - 12), f"{tick}%", font=f_reg(18), fill=MUTED)

    groups = len(labels)
    group_w = width / groups
    bar_w = 38
    for i, label in enumerate(labels):
        gx = px1 + i * group_w + group_w / 2
        bx1 = gx - 28 - bar_w
        bx2 = gx - 10
        ax1 = gx + 10
        ax2 = gx + 10 + bar_w
        before_h = y_pos(before[i] * 100)
        after_h = y_pos(after[i] * 100)

        rr(draw, (bx1, before_h, bx2, py2), GREY_SOFT, radius=16)
        rr(draw, (ax1, after_h, ax2, py2), ACCENT if i == 1 else GREY, radius=16)

        wrapped = label.split("\n")
        draw.text((gx - 58, py2 + 22), wrapped[0], font=f_reg(16), fill=MUTED)
        draw.text((gx - 48, py2 + 42), wrapped[1], font=f_reg(16), fill=MUTED)

        if i == 1:
            drop_pp = (after[i] - before[i]) * 100
            note = f"{drop_pp:.1f} п.п."
            rr(draw, (gx - 78, after_h - 74, gx + 84, after_h - 26), "#fff8e7", outline="#efcf7b", width=2, radius=18)
            draw.text((gx - 56, after_h - 61), note, font=f_bold(20), fill="#b45309")

    legend_y = y1 + 20
    legend_x1 = x2 - 246
    legend_x2 = x2 - 110
    rr(draw, (legend_x1, legend_y, legend_x1 + 30, legend_y + 22), GREY_SOFT, radius=11)
    draw.text((legend_x1 + 42, legend_y - 2), "Недели 1–6", font=f_reg(18), fill=MUTED)
    rr(draw, (legend_x2, legend_y, legend_x2 + 30, legend_y + 22), ACCENT, radius=11)
    draw.text((legend_x2 + 42, legend_y - 2), "Недели 7–8", font=f_reg(18), fill=MUTED)


def make_png():
    data = load_metrics()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = Image.new("RGBA", (W, H), BG)
    base = add_blob(base, (-80, -80, 540, 340), (245, 158, 11, 32))
    base = add_blob(base, (1650, -120, 2240, 360), (14, 165, 164, 26))
    base = add_blob(base, (1500, 920, 2200, 1500), (17, 24, 39, 18))
    base = base.convert("RGB")
    draw = ImageDraw.Draw(base)

    draw.text((90, 72), "1 задание • бизнес-кейс", font=f_bold(20), fill="#a16207")
    draw.text((90, 120), "После изменений в поиске просел ключевой шаг воронки", font=f_black(52), fill=INK)
    draw.text((90, 190), "Регистрация стабильна, проблема локализована после неё — между регистрацией и открытием поиска.", font=f_reg(27), fill=TEXT)

    pill(draw, 90, 256, 300, "Установки", f"+{data['installs_growth'] * 100:.1f}%", "#dcfce7")
    pill(draw, 414, 256, 350, "Общая конверсия", f"{data['overall_before'] * 100:.1f}% → {data['overall_after'] * 100:.1f}%", "#ede9fe")
    pill(draw, 788, 256, 380, "Ключевой провал", f"{data['reg_search_before'] * 100:.1f}% → {data['reg_search_after'] * 100:.1f}%", "#ffedd5")

    left_card = (70, 410, 1085, 900)
    right_card = (1115, 410, 2130, 900)
    summary_card = (70, 940, 2130, 1315)

    for card in [left_card, right_card, summary_card]:
        base = shadow_card(base.convert("RGBA"), card, radius=34, blur=22, offset=(0, 10)).convert("RGB")
        draw = ImageDraw.Draw(base)
        rr(draw, card, CARD, outline=LINE, width=2, radius=34)

    draw_line_chart(draw, data["weekly"], left_card)
    draw_bar_chart(draw, data["step_labels"], data["step_before"], data["step_after"], right_card)

    draw.text((110, 980), "Вывод", font=f_bold(28), fill=INK)
    multiline(
        draw,
        (110, 1025),
        "Основной отток появляется сразу после регистрации: пользователи доходят до аккаунта, но заметно реже открывают поиск авто. Остальные шаги воронки остаются около прежних значений, поэтому root cause вероятнее всего в новом опыте поиска.",
        f_reg(25),
        TEXT,
        1220,
        spacing=8,
    )

    draw.text((1370, 980), "Что проверить дальше", font=f_bold(28), fill=INK)
    bullets = [
        "Разбить шаг «регистрация → поиск» по версии приложения, OS, городу и типу устройства.",
        "Проверить latency, ошибки, пустые результаты и разрешение геолокации в новом поиске.",
        "Посмотреть event-path после регистрации: куда уходит пользователь, если не доходит до поиска.",
    ]
    y = 1030
    for bullet in bullets:
        draw.ellipse((1372, y + 10, 1386, y + 24), fill=SUCCESS)
        wrapped = multiline(draw, (1405, y), bullet, f_reg(23), TEXT, 650, spacing=8)
        y += (wrapped.count("\n") + 1) * 31 + 20

    base.save(PNG_PATH, format="PNG", optimize=True)


def make_pdf():
    page_w, page_h = landscape(A4)
    pdf = canvas.Canvas(str(PDF_PATH), pagesize=(page_w, page_h))
    pdf.drawImage(ImageReader(str(PNG_PATH)), 0, 0, width=page_w, height=page_h, preserveAspectRatio=False, mask="auto")
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    make_png()
    make_pdf()
