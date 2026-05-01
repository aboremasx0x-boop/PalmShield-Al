import os
import tempfile
import io
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

MODEL_PATH = "models/palmshield_model.joblib"
BASELINE_FILE = "baseline.csv"
HISTORY_FILE = "history.csv"
DATASET_URL = "https://drive.google.com/drive/folders/1HCG8jf-_aqv8nvvyoevufK-xAeb6rXlk?usp=sharing"


def ar(text):
    return get_display(arabic_reshaper.reshape(str(text)))


def create_pdf_report(palm_id, result, confidence, risk, recommendation, baseline_delta):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    font_path = "C:/Windows/Fonts/arial.ttf"
    pdfmetrics.registerFont(TTFont("ArabicFont", font_path))

    style = ParagraphStyle(
        name="ArabicStyle",
        fontName="ArabicFont",
        fontSize=12,
        leading=20,
        alignment=2
    )

    title_style = ParagraphStyle(
        name="TitleStyle",
        fontName="ArabicFont",
        fontSize=18,
        leading=24,
        alignment=1
    )

    content = []

    content.append(Paragraph("PalmShield AI Report", title_style))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Early Acoustic Detection of Red Palm Weevil", title_style))
    content.append(Spacer(1, 24))

    content.append(Paragraph(ar(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), style))
    content.append(Paragraph(ar(f"رقم النخلة: {palm_id}"), style))
    content.append(Paragraph(ar(f"النتيجة: {result}"), style))
    content.append(Paragraph(ar(f"نسبة الثقة: {confidence:.1f}%"), style))
    content.append(Paragraph(ar(f"مستوى الخطورة: {risk}"), style))
    content.append(Paragraph(ar(f"التغير عن المرجع: {baseline_delta}"), style))

    content.append(Spacer(1, 20))

    content.append(Paragraph(ar("التوصية الذكية:"), style))
    content.append(Paragraph(ar(recommendation), style))

    content.append(Spacer(1, 20))

    content.append(Paragraph(ar("ملخص التشخيص:"), style))
    content.append(Paragraph(
        ar("يعتمد هذا التقرير على تحليل صوتي بالذكاء الاصطناعي لتقدير احتمالية إصابة النخلة بسوسة النخيل الحمراء."),
        style
    ))

    content.append(Spacer(1, 20))

    content.append(Paragraph(ar("ملاحظة مهمة:"), style))
    content.append(Paragraph(
        ar("هذا التقرير صادر من نموذج أولي للعرض، ويجب تأكيد النتائج لاحقًا بفحص ميداني وتسجيلات صوتية حقيقية."),
        style
    ))

    doc.build(content)
    buffer.seek(0)
    return buffer


def save_history(palm_id, result, confidence, risk, baseline_delta):
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "palm_id": palm_id,
        "result": result,
        "confidence": round(confidence, 2),
        "risk": risk,
        "baseline_delta": baseline_delta
    }

    df = pd.DataFrame([row])
    df.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not os.path.exists(HISTORY_FILE),
        index=False,
        encoding="utf-8-sig"
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


def calculate_baseline_delta(current_features, baseline_features):
    if baseline_features is None:
        return "لا يوجد مرجع"

    delta = float(np.linalg.norm(current_features - baseline_features))
    return round(delta, 4)


st.set_page_config(
    page_title="PalmShield AI",
    page_icon="🌴",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
}
.card {
    padding: 22px;
    border-radius: 18px;
    background-color: #f7f9fb;
    border: 1px solid #e6e6e6;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("PalmShield AI")

st.sidebar.markdown("### 📂 Dataset")
st.sidebar.markdown(
    f"[⬇️ تحميل بيانات التجربة من Google Drive]({DATASET_URL})"
)

lang = st.sidebar.selectbox("Language / اللغة", ["العربية", "English"])

if lang == "العربية":
    subtitle = "نظام ذكي للكشف المبكر عن سوسة النخيل الحمراء بالصوت"
    intro = "يرفع المستخدم تسجيلًا أو عدة تسجيلات صوتية للنخلة، ويحلل النظام النمط الصوتي ويقارنه بالمرجع الخاص بالنخلة عند توفره."
    upload_label = "ارفع ملف أو عدة ملفات صوتية WAV"
    demo_btn = "تشغيل تجربة جاهزة"
    healthy = "نخلة سليمة"
    infected = "اشتباه إصابة بسوسة النخيل"
    confidence_label = "نسبة الثقة"
    recommendation_label = "التوصية الذكية"
    spectrogram_label = "التحليل الطيفي للصوت"
else:
    subtitle = "Early Acoustic Detection of Red Palm Weevil"
    intro = "Upload one or multiple palm-trunk audio recordings. The system analyzes the acoustic pattern and compares it with the palm baseline when available."
    upload_label = "Upload one or more WAV files"
    demo_btn = "Run Demo"
    healthy = "Healthy Palm"
    infected = "Suspected RPW Infestation"
    confidence_label = "Confidence"
    recommendation_label = "Smart Recommendation"
    spectrogram_label = "Audio Spectrogram"

st.markdown("<div class='main-title'>PalmShield AI</div>", unsafe_allow_html=True)
st.subheader(subtitle)
st.write(intro)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. تأكد من رفع ملف النموذج داخل models/palmshield_model.joblib")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Input / الإدخال")
    palm_id = st.text_input("Palm ID / رقم النخلة", value="Palm-001")

    uploaded_files = st.file_uploader(
        upload_label,
        type=["wav"],
        accept_multiple_files=True
    )

    demo_clicked = st.button(demo_btn)

    if demo_clicked:
        demo_paths = [
            "data/infested/infested_0001.wav",
            "data/infested/infested_0002.wav",
            "data/infested/infested_0003.wav"
        ]
        uploaded_files = [open(p, "rb") for p in demo_paths if os.path.exists(p)]

        if not uploaded_files:
            st.warning("ملفات التجربة غير موجودة داخل التطبيق. يمكنك تحميلها من رابط Google Drive في القائمة الجانبية ثم رفع ملف WAV يدويًا.")

with col2:
    st.markdown("### Project Value / قيمة المشروع")
    st.markdown("""
    <div class='card'>
    <b>Innovation:</b> Bioacoustics + AI + Baseline Tracking<br>
    <b>New Feature:</b> Multi-recording voting<br>
    <b>Dataset:</b> External Google Drive sample data<br>
    <b>Impact:</b> Early detection, lower pesticide use, palm protection
    </div>
    """, unsafe_allow_html=True)

if uploaded_files:
    st.divider()
    st.markdown("### Audio Preview / معاينة الصوت")

    all_probs = []
    all_features = []
    temp_paths = []

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
    pred = 1 if avg_prob >= 0.5 else 0

    baseline_features = load_baseline(palm_id)
    baseline_delta = calculate_baseline_delta(final_features, baseline_features)

    st.divider()
    st.markdown("## Result / النتيجة")

    r1, r2, r3, r4 = st.columns(4)

    if pred == 1:
        result_text = infected

        if confidence >= 90:
            risk = "خطر عالي"
            recommendation = "يلزم فحص ميداني فوري للنخلة، وإعادة التسجيل الصوتي، ثم تطبيق برنامج مكافحة متكاملة معتمد عند تأكيد الإصابة."
        elif confidence >= 70:
            risk = "خطر متوسط"
            recommendation = "ينصح بإعادة فحص النخلة خلال 48 ساعة ومقارنة التسجيل الجديد بالتسجيل الحالي."
        else:
            risk = "اشتباه منخفض"
            recommendation = "ينصح بالمراقبة المستمرة وإعادة التسجيل لاحقًا."

        with r1:
            st.error(result_text)
    else:
        result_text = healthy
        risk = "آمن"
        recommendation = "ينصح بالاستمرار في المتابعة الدورية وإعادة الفحص الصوتي بشكل منتظم."

        with r1:
            st.success(result_text)

    with r2:
        st.metric(confidence_label, f"{confidence:.1f}%")

    with r3:
        st.info(risk)

    with r4:
        st.metric("Baseline Change", baseline_delta)

    st.markdown(f"### {recommendation_label}")
    st.write(recommendation)

    st.divider()

    st.markdown("## Baseline / المرجع الصوتي")
    if baseline_features is None:
        st.warning("لا يوجد مرجع صوتي لهذه النخلة. يمكنك حفظ التسجيل الحالي كمرجع إذا كانت النخلة سليمة.")
    else:
        st.success("تمت مقارنة التسجيل الحالي مع المرجع الصوتي المحفوظ لهذه النخلة.")

    if st.button("Save Current Recording as Baseline / حفظ التسجيل الحالي كمرجع"):
        save_baseline(palm_id, final_features)
        st.success("تم حفظ المرجع الصوتي لهذه النخلة بنجاح.")

    st.divider()

    st.markdown("## Palm Health Report / تقرير حالة النخلة")

    if pred == 1:
        st.markdown("""
        **Diagnosis / التشخيص:**  
        النمط الصوتي يشير إلى نشاط داخلي قد يتوافق مع تغذية يرقات سوسة النخيل الحمراء.

        **Visual Interpretation / التفسير البصري:**  
        قد يظهر في التحليل الطيفي نمط نبضي غير طبيعي مقارنة بالنخلة السليمة.

        **Impact / الأثر:**  
        قد يكون الضرر الداخلي في الجذع بدأ قبل ظهور الأعراض الخارجية.

        **Action Plan / خطة العمل:**  
        - فحص ميداني فوري  
        - إعادة التسجيل الصوتي  
        - تطبيق مكافحة متكاملة عند تأكيد الإصابة  
        """)
    else:
        st.markdown("""
        **Diagnosis / التشخيص:**  
        لم يتم رصد نشاط صوتي داخلي غير طبيعي في هذه العينة.

        **Visual Interpretation / التفسير البصري:**  
        التحليل الطيفي أقرب إلى الاهتزازات الطبيعية أو الضوضاء الخلفية.

        **Impact / الأثر:**  
        حالة النخلة مصنفة كآمنة بناءً على هذه العينة الصوتية.

        **Action Plan / خطة العمل:**  
        - الاستمرار في المتابعة  
        - إعادة الفحص دوريًا  
        - مقارنة التسجيلات المستقبلية  
        """)

    save_history(palm_id, result_text, confidence, risk, baseline_delta)

    pdf_report = create_pdf_report(
        palm_id=palm_id,
        result=result_text,
        confidence=confidence,
        risk=risk,
        recommendation=recommendation,
        baseline_delta=baseline_delta
    )

    st.download_button(
        label="Download PDF Report / تحميل التقرير PDF",
        data=pdf_report,
        file_name=f"PalmShield_Report_{palm_id}.pdf",
        mime="application/pdf"
    )

    st.divider()

    sr, x = wavfile.read(temp_paths[0])
    if x.ndim > 1:
        x = x.mean(axis=1)

    f, t, Sxx = spectrogram(x, fs=sr)

    st.markdown(f"### {spectrogram_label}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t, f, Sxx, shading="auto")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    st.pyplot(fig)

    st.divider()

    st.markdown("## History Dashboard / سجل الفحوصات")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        st.dataframe(history_df.tail(20), use_container_width=True)

    st.markdown("### Hackathon Pitch / ملخص العرض")
    st.info(
        "PalmShield AI is not just a one-time detector. It tracks each palm over time using baseline comparison, multi-recording voting, automated reports, and external sample data access."
    )

    for p in temp_paths:
        try:
            os.remove(p)
        except:
            pass
