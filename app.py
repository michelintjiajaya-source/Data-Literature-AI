import streamlit as st
import os
os.environ['HF_HOME'] ='/tmp'
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

st.set_page_config(page_title="Data Literature AI", layout="centered")

desain_premium = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1507842217343-583bb7270b66?q=80&w=2000&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
}
.block-container {
    background-color: rgba(20, 20, 20, 0.85);
    padding: 3rem;
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.7);
    margin-top: 2rem;
    margin-bottom: 2rem;
}
h1 {
    font-family: 'Georgia', serif;
    color: #ffffff !important;
    text-align: center;
    font-size: 3rem !important;
    margin-bottom: 0.5rem;
}
p {
    color: #e0e0e0 !important;
    text-align: center;
    font-size: 1.1rem;
}
.eval-box {
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #cccccc;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(desain_premium, unsafe_allow_html=True)

# KUNCI RAHASIA DIAMANKAN DI SINI MENGGUNAKAN FITUR SECRETS
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def siapkan_sistem_rag(pilihan_mesin):
    lokasi_dokumen = "katalog_buku.pdf"
    pemuat_dokumen = PyPDFLoader(lokasi_dokumen)
    dokumen_mentah = pemuat_dokumen.load()

    pemotong_teks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    halaman_dokumen = pemotong_teks.split_documents(dokumen_mentah)

    for halaman in halaman_dokumen:
        halaman.page_content = halaman.page_content.replace('â€œ', '"').replace('â€', '"').replace('â€™', "'")

    pembungkus_vektor = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder="/tmp"
    )
    basis_data_vektor = FAISS.from_documents(halaman_dokumen, pembungkus_vektor)

    model_bahasa = ChatGroq(model_name=pilihan_mesin, temperature=0)
    return basis_data_vektor, model_bahasa


def evaluasi_jawaban(pertanyaan, jawaban, konteks, llm):
    prompt_evaluasi = f"""
Kamu adalah evaluator sistem RAG. Berikan penilaian untuk jawaban chatbot berikut.

Pertanyaan pengguna: {pertanyaan}

Konteks dokumen yang tersedia: {konteks[:1500]}

Jawaban chatbot: {jawaban}

Berikan skor 1-5 untuk tiga metrik berikut. Jawab HANYA dalam format JSON seperti ini, tanpa teks lain:
{{
  "relevance": <skor 1-5>,
  "relevance_reason": "<alasan singkat>",
  "faithfulness": <skor 1-5>,
  "faithfulness_reason": "<alasan singkat>",
  "completeness": <skor 1-5>,
  "completeness_reason": "<alasan singkat>"
}}

Keterangan metrik:
- relevance: seberapa relevan jawaban dengan pertanyaan (1=tidak relevan, 5=sangat relevan)
- faithfulness: seberapa setia jawaban pada konteks dokumen, tidak mengarang (1=banyak halusinasi, 5=sangat setia)
- completeness: seberapa lengkap jawaban menjawab pertanyaan (1=sangat kurang, 5=sangat lengkap)
"""
    try:
        hasil = llm.invoke(prompt_evaluasi)
        teks = hasil.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(teks)
    except Exception:
        return None


def tampilkan_evaluasi(data_eval):
    if not data_eval:
        return

    def bar_skor(skor):
        terisi = "[" + "#" * int(skor) + "-" * (5 - int(skor)) + "]"
        return f"{terisi} {skor}/5"

    st.markdown(f"""
<div class="eval-box">
<strong>[ LLM Evaluation Result ]</strong><br><br>
Relevance&nbsp;&nbsp;&nbsp;&nbsp;: {bar_skor(data_eval.get('relevance', 0))} - {data_eval.get('relevance_reason', '')}<br>
Faithfulness : {bar_skor(data_eval.get('faithfulness', 0))} - {data_eval.get('faithfulness_reason', '')}<br>
Completeness : {bar_skor(data_eval.get('completeness', 0))} - {data_eval.get('completeness_reason', '')}
</div>
""", unsafe_allow_html=True)


# UI 

st.markdown("<h1>Data Literature Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p>Discover your data analytics books through natural conversation.</p>", unsafe_allow_html=True)

kolom_kiri, kolom_tengah, kolom_kanan = st.columns([1, 2, 1])
with kolom_tengah:
    mesin_ai = st.selectbox(
        "Select AI Engine:",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0
    )

tampilkan_eval = st.toggle("Show Evaluation", value=True)
st.write("---")

if "riwayat_obrolan" not in st.session_state:
    st.session_state.riwayat_obrolan = []

if "riwayat_evaluasi" not in st.session_state:
    st.session_state.riwayat_evaluasi = []

try:
    vektor_db, llm = siapkan_sistem_rag(mesin_ai)
    pencari_dokumen = vektor_db.as_retriever(search_kwargs={"k": 10})

    sistem_prompt = (
        "Anda adalah Asisten Katalog Buku yang ramah dan suportif. "
        "Tugas Anda membantu pengguna menemukan buku berdasarkan <konteks> yang diberikan. "
        "1. Anda boleh fleksibel: Jika pengguna mencari 'belajar Excel', Anda boleh merekomendasikan buku 'Microsoft Excel' yang ada di konteks. "
        "2. ATURAN MUTLAK: Anda HANYA boleh menyebutkan judul dan penulis yang benar-benar ada di <konteks>. Dilarang mengarang buku. "
        "3. ATURAN MUTLAK: DILARANG KERAS menyarankan pengguna mencari di Amazon, Google, internet, atau toko lain. Anda tidak punya akses ke sana. "
        "4. Jika topik yang dicari benar-benar tidak ada di konteks, jawab dengan ramah: 'Maaf ya, sepertinya buku dengan topik tersebut belum tersedia di katalog kita saat ini.' lalu hentikan jawaban Anda tanpa memberi saran tambahan."
        "Konteks: {context}"
    )

    prompt_utama = ChatPromptTemplate.from_messages([
        ("system", sistem_prompt),
        MessagesPlaceholder(variable_name="riwayat_obrolan"),
        ("human", "{input}"),
    ])

    rantai_tanya_jawab = create_stuff_documents_chain(llm, prompt_utama)
    rantai_rag_memori = create_retrieval_chain(pencari_dokumen, rantai_tanya_jawab)

    for i, pesan in enumerate(st.session_state.riwayat_obrolan):
        if isinstance(pesan, HumanMessage):
            st.chat_message("user").write(pesan.content)
        elif isinstance(pesan, AIMessage):
            st.chat_message("assistant").write(pesan.content)
            idx_eval = i // 2
            if tampilkan_eval and idx_eval < len(st.session_state.riwayat_evaluasi):
                tampilkan_evaluasi(st.session_state.riwayat_evaluasi[idx_eval])

    pertanyaan_pengguna = st.chat_input("Type a book title or topic here...")

    if pertanyaan_pengguna:
        st.chat_message("user").write(pertanyaan_pengguna)
        st.session_state.riwayat_obrolan.append(HumanMessage(content=pertanyaan_pengguna))

        with st.spinner("Analyzing literature collection..."):
            jawaban = rantai_rag_memori.invoke({
                "input": pertanyaan_pengguna,
                "riwayat_obrolan": st.session_state.riwayat_obrolan
            })
            hasil_teks = jawaban["answer"]
            konteks_dokumen = " ".join([doc.page_content for doc in jawaban.get("context", [])])
            st.chat_message("assistant").write(hasil_teks)
            st.session_state.riwayat_obrolan.append(AIMessage(content=hasil_teks))

        if tampilkan_eval:
            with st.spinner("Evaluating response..."):
                hasil_eval = evaluasi_jawaban(pertanyaan_pengguna, hasil_teks, konteks_dokumen, llm)
                st.session_state.riwayat_evaluasi.append(hasil_eval)
                tampilkan_evaluasi(hasil_eval)
        else:
            st.session_state.riwayat_evaluasi.append(None)

except Exception as e:
    st.error(f"Sistem sedang mengalami penyesuaian: {e}")