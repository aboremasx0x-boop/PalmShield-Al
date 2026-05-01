# ================================
# PalmShield AI – Final Winning Version
# ================================

import os
import io
import tempfile
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import spectrogram

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import arabic_reshaper
from bidi.algorithm import get_display

from features import extract_features


# ================================
# Paths
# ================================
MODEL_PATH = "models/palmshield_model.joblib"
BASELINE_FILE = "baseline.csv"
HISTORY_FILE = "history.csv"

DATASET_URL = "https://drive.google.com/drive/folders/1HCG8jf-_aqv8nvvyoevufK-xAeb6rXlk?usp=sharing"


# ================================
# Arabic / PDF Helpers
# ================================
def ar(text):
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)


def get_font():
    paths = [
        "fonts/Amiri-Regular.ttf",
        "Amiri-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def create_pdf_report(palm_id, result, confidence, risk, recommendation, delta):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    font_path = get_font()

    if font_path:
        pdfmetrics.registerFont(TTFont("PalmFont", font_path))
        font_name = "PalmFont"
        fmt = lambda x: ar(x)
    else:
        font_name = "Helvetica"
        fmt = lambda x: str(x)

    style = ParagraphStyle(
        name="ReportStyle",
        fontName=font_name,
        fontSize=12,
        leading=20,
        alignment=2,
    )

    title_style = ParagraphStyle(
        name="TitleStyle",
        fontName=font_name,
        fontSize=18,
        leading=24,
        alignment=1,
    )

    content = [
        Paragraph("PalmShield AI Report", title_style),
        Spacer(1, 12),
        Paragraph("Early Acoustic Detection of Red Palm Weevil", title_style),
        Spacer(1, 24),
        Paragraph(fmt(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), style),
        Paragraph(fmt(f"رقم النخلة: {palm_id}"), style),
        Paragraph(fmt(f"النتيجة: {result}"), style),
        Paragraph(fmt(f"نسبة الثقة: {confidence:.1f}%"), style),
        Paragraph(fmt(f"مستوى الخطورة: {risk}"), style),
        Paragraph(fmt(f"التغير عن المرجع: {delta}"), style),
        Spacer(1, 18),
        Paragraph(fmt("التوصية الذكية:"), style),
        Paragraph(fmt(recommendation), style),
        Spacer(1, 18),
        Paragraph(fmt("ملخص:"), style),
        Paragraph(
            fmt("يعتمد PalmShield AI على تحليل الإشارات الصوتية داخل جذع النخلة للكشف المبكر عن احتمالية الإصابة بسوسة النخيل الحمراء."),
            style,
        ),
        Spacer(1, 18),
        Paragraph(fmt("ملاحظة:"), style),
        Paragraph(
            fmt("هذا التقرير صادر من نموذج أولي قابل للتطوير، ويجب تأكيد النتائج لاحقًا بفحص ميداني وتسجيلات صوتية حقيقية."),
            style,
        ),
    ]

    doc.build(content)
    buffer.seek(0)
    return buffer


# ================================
# Data Helpers
# ================================
def save_history(palm_id, result, confidence, risk, delta):
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "palm_id": palm_id,
        "result": result,
        "confidence": round(confidence, 2),
        "risk": risk,
        "baseline_delta": delta,
    }

    df = pd.DataFrame([row])
    df.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not os.path.exists(HISTORY_FILE),
        index=False,
        encoding="utf-8-sig",
    )


def save_baseline(palm_id, features_vector):
    data = {"palm_id": palm_id}

    for i, value in enumerate(features_vector):
        data[f"f{i}"] = value

    new_row = pd.DataFrame([data])

    if os.path.exists(BASELINE_FILE):
        df = pd.read_csv(BASELINE_FILE)
        df = df[df["palm_id"] != palm_id]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(BASELINE_FILE, index=False, encoding="utf-8-sig")


def load_baseline(palm_id):
    if not os.path.exists(BASELINE_FILE):
        return None

    df = pd.read_csv(BASELINE_FILE)
    row = df[df["palm_id"] == palm_id]

    if row.empty:
        return None

    feature_cols = [c for c in df.columns if c.startswith("f")]
    return row.iloc[0][feature_cols].values.astype(float)


def calculate_delta(current_features, baseline_features):
    if baseline_features is None:
        return "لا يوجد مرجع"

    return round(float(np.linalg.norm(current_features - baseline_features)), 4)


def classify_result(avg_prob, confidence):
    pred = 1 if avg_prob >= 0.5 else 0

    if pred == 1:
        result = "اشتباه إصابة بسوسة النخيل"
        if confidence >= 90:
            risk = "خطر عالي"
            recommendation = "يلزم فحص ميداني فوري للنخلة، وإعادة التسجيل الصوتي، ثم تطبيق برنامج مكافحة متكاملة معتمد عند تأكيد الإصابة."
        elif confidence >= 70:
            risk = "خطر متوسط"
            recommendation = "ينصح بإعادة الفحص خلال 48 ساعة ومقارنة التسجيل الجديد بالتسجيل الحالي."
        else:
            risk = "اشتباه منخفض"
            recommendation = "ينصح بالمراقبة المستمرة وإعادة التسجيل لاحقًا."
    else:
        result = "نخلة سليمة"
        risk = "آمن"
        recommendation = "ينصح بالاستمرار في المتابعة الدورية وإعادة الفحص الصوتي بشكل منتظم."

    return pred, result, risk, recommendation


# ================================
# UI Setup
# ================================
st.set_page_config(
    page_title="PalmShield AI",
    page_icon="🌴",
    layout="wide",
)

st.markdown(
    """
<style>
.hero {
    background: linear-gradient(135deg, #0b3d2e, #168a52);
    padding: 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 22px;
}
.hero h1 {
    font-size: 46px;
    margin-bottom: 8px;
}
.hero p {
    font-size: 20px;
}
.card {
    background: #f8faf9;
    border: 1px solid #e5e7eb;
    padding: 22px;
    border-radius: 18px;
    min-height: 125px;
}
.small-card {
    background: white;
    border: 1px solid #e5e7eb;
    padding: 16px;
    border-radius: 14px;
}
.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 22px;
}
.footer {
    background: #f8faf9;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
}
</style>
""",
    unsafe_allow_html=True,
)


# ================================
# Sidebar
# ================================
st.sidebar.title("PalmShield AI")
st.sidebar.markdown("### روابط المشروع")
st.sidebar.markdown(f"[تحميل بيانات التجربة من Google Drive]({DATASET_URL})")

st.sidebar.markdown("### وضع النظام")
field_mode = st.sidebar.toggle("Field Mode / الوضع الميداني", value=True)

st.sidebar.markdown("### فكرة المشروع")
st.sidebar.info(
    "نظام ذكي للكشف المبكر عن سوسة النخيل الحمراء باستخدام الصوت الحيوي والذكاء الاصطناعي."
)


# ================================
# Hero
# ================================
st.markdown(
    """
<div class="hero">
<h1>PalmShield AI</h1>
<p>نظام إنذار مبكر للكشف عن سوسة النخيل الحمراء بالصوت الحيوي والذكاء الاصطناعي قبل ظهور الأعراض الخارجية</p>
</div>
""",
    unsafe_allow_html=True,
)


# ================================
# Load Model
# ================================
if not os.path.exists(MODEL_PATH):
    st.error("لم يتم العثور على النموذج. تأكد من رفع الملف: models/palmshield_model.joblib")
    st.stop()

model = joblib.load(MODEL_PATH)


# ================================
# Executive Dashboard
# ================================
st.markdown('<div class="section-title">Executive Dashboard / لوحة القرار</div>', unsafe_allow_html=True)

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.metric("Target Pest", "Red Palm Weevil")

with d2:
    st.metric("Detection Type", "Acoustic AI")

with d3:
    st.metric("System Mode", "Prototype")

with d4:
    st.metric("Report", "PDF Ready")


# ================================
# Input Section
# ================================
st.markdown('<div class="section-title">1) رفع التسجيل الصوتي</div>', unsafe_allow_html=True)

col_input, col_value = st.columns([1, 1])

with col_input:
    palm_id = st.text_input("Palm ID / رقم النخلة", value="Palm-001")
    uploaded_files = st.file_uploader(
        "ارفع ملفًا أو عدة ملفات WAV",
        type=["wav"],
        accept_multiple_files=True,
    )

with col_value:
    st.markdown(
        """
<div class="card">
<b>قيمة الابتكار:</b><br>
- كشف مبكر قبل ظهور الأعراض<br>
- تحليل عدة تسجيلات وليس تسجيلًا واحدًا فقط<br>
- مقارنة مع Baseline لكل نخلة<br>
- تقرير PDF قابل للاستخدام الميداني<br>
- قابل للتحول إلى جهاز IoT منخفض التكلفة
</div>
""",
        unsafe_allow_html=True,
    )


# ================================
# Main Analysis
# ================================
analysis_done = False

if uploaded_files:
    all_probs = []
    all_features = []
    temp_paths = []

    st.markdown('<div class="section-title">2) معاينة التسجيلات</div>', unsafe_allow_html=True)

    for uploaded in uploaded_files:
        st.audio(uploaded, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
            temp_paths.append(tmp_path)

        feature_vector = extract_features(tmp_path)
        all_features.append(feature_vector)

        prob = model.predict_proba(feature_vector.reshape(1, -1))[0][1]
        all_probs.append(prob)

    avg_prob = float(np.mean(all_probs))
    confidence = max(avg_prob, 1 - avg_prob) * 100
    final_features = np.mean(np.vstack(all_features), axis=0)

    baseline_features = load_baseline(palm_id)
    delta = calculate_delta(final_features, baseline_features)

    pred, result, risk, recommendation = classify_result(avg_prob, confidence)

    analysis_done = True

    save_history(palm_id, result, confidence, risk, delta)

    # ================================
    # Result Section
    # ================================
    st.markdown('<div class="section-title">3) نتيجة التحليل</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        if pred == 1:
            st.error(result)
        else:
            st.success(result)

    with r2:
        st.metric("نسبة الثقة", f"{confidence:.1f}%")

    with r3:
        st.metric("مستوى الخطورة", risk)

    with r4:
        st.metric("Baseline Change", delta)

    st.markdown("### التوصية الذكية")
    st.info(recommendation)

    # ================================
    # IoT Simulation
    # ================================
    st.markdown('<div class="section-title">4) محاكاة جهاز IoT ميداني</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(
            """
<div class="small-card">
<b>Sensor</b><br>
حساس صوتي مثبت على جذع النخلة
</div>
""",
            unsafe_allow_html=True,
        )

    with i2:
        st.markdown(
            """
<div class="small-card">
<b>AI Node</b><br>
تحليل الإشارة واستخراج الخصائص
</div>
""",
            unsafe_allow_html=True,
        )

    with i3:
        st.markdown(
            """
<div class="small-card">
<b>Decision</b><br>
تنبيه + تقرير + توصية ميدانية
</div>
""",
            unsafe_allow_html=True,
        )

    # ================================
    # Baseline Section
    # ================================
    st.markdown('<div class="section-title">5) المرجع الصوتي للنخلة</div>', unsafe_allow_html=True)

    if baseline_features is None:
        st.warning("لا يوجد مرجع صوتي لهذه النخلة. إذا كانت النخلة سليمة، يمكنك حفظ التسجيل الحالي كمرجع.")
    else:
        st.success("تمت مقارنة التسجيل الحالي مع المرجع الصوتي المحفوظ لهذه النخلة.")

    if st.button("حفظ التسجيل الحالي كمرجع Baseline"):
        save_baseline(palm_id, final_features)
        st.success("تم حفظ المرجع الصوتي للنخلة بنجاح.")

    # ================================
    # Spectrogram
    # ================================
    st.markdown('<div class="section-title">6) التحليل الطيفي للصوت</div>', unsafe_allow_html=True)

    try:
        sr, x = wavfile.read(temp_paths[0])
        if x.ndim > 1:
            x = x.mean(axis=1)

        f, t, Sxx = spectrogram(x, fs=sr)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.pcolormesh(t, f, Sxx, shading="auto")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [sec]")
        ax.set_title("Audio Spectrogram")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"تعذر عرض التحليل الطيفي: {e}")

    # ================================
    # Field Map Simulation
    # ================================
    st.markdown('<div class="section-title">7) خريطة مزرعة تجريبية</div>', unsafe_allow_html=True)

    map_df = pd.DataFrame(
        {
            "lat": [21.4858, 21.4862, 21.4866, 21.4870],
            "lon": [39.1925, 39.1930, 39.1935, 39.1940],
        }
    )

    st.map(map_df)

    st.caption("الخريطة أعلاه محاكاة ميدانية توضح كيف يمكن ربط النظام لاحقًا بمواقع النخيل في المزرعة.")

    # ================================
    # PDF Report
    # ================================
    st.markdown('<div class="section-title">8) التقرير الميداني</div>', unsafe_allow_html=True)

    try:
        pdf_report = create_pdf_report(
            palm_id=palm_id,
            result=result,
            confidence=confidence,
            risk=risk,
            recommendation=recommendation,
            delta=delta,
        )

        st.download_button(
            label="تحميل تقرير PDF",
            data=pdf_report,
            file_name=f"PalmShield_Report_{palm_id}.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.warning("تعذر إنشاء التقرير PDF على الخادم الحالي.")
        st.text(str(e))

    for p in temp_paths:
        try:
            os.remove(p)
        except Exception:
            pass


# ================================
# History Dashboard
# ================================
st.markdown('<div class="section-title">9) سجل الفحوصات</div>', unsafe_allow_html=True)

h1, h2 = st.columns([3, 1])

with h1:
    st.markdown("### آخر الفحوصات")

with h2:
    if st.button("إعادة إدخال البيانات"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("تم مسح سجل الفحوصات.")
        st.rerun()

if os.path.exists(HISTORY_FILE):
    try:
        history_df = pd.read_csv(HISTORY_FILE)
        if not history_df.empty:
            st.dataframe(history_df.tail(20), use_container_width=True)

            total_scans = len(history_df)
            suspected = history_df["result"].astype(str).str.contains("اشتباه").sum()
            safe = total_scans - suspected

            s1, s2, s3 = st.columns(3)
            s1.metric("عدد الفحوصات", total_scans)
            s2.metric("اشتباه إصابة", suspected)
            s3.metric("سليمة", safe)
        else:
            st.info("لا يوجد سجل فحوصات حتى الآن.")
    except Exception:
        st.info("لا يوجد سجل فحوصات صالح حتى الآن.")
else:
    st.info("لا يوجد سجل فحوصات حتى الآن.")


# ================================
# Pitch Section
# ================================
st.markdown('<div class="section-title">Hackathon Pitch / ملخص العرض</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="footer">
<b>PalmShield AI</b> ليس مجرد نموذج تشخيص، بل منصة إنذار مبكر قابلة للتحول إلى جهاز ميداني منخفض التكلفة لحماية نخيل المملكة من سوسة النخيل الحمراء.<br><br>
<b>القيمة:</b> كشف مبكر، تقليل مبيدات، حماية الإنتاج، دعم الأمن الغذائي، وقابلية التوسع للمزارع والجهات الحكومية.
</div>
""",
    unsafe_allow_html=True,
)
