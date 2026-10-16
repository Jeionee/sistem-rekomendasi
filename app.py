import os
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# =====================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="SPK Rekomendasi Reviewer Jurnal - Multilingual SBERT",
    layout="wide",
)

# =====================================================================
# AUTO-GENERATOR DATASET CSV (JIKA FILE BELUM ADA DI FOLDER)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MS_PATH = os.path.join(BASE_DIR, "manuscripts.csv")
PUB_PATH = os.path.join(BASE_DIR, "reviewer_publications.csv")

DEFAULT_MANUSCRIPTS_CSV = """manuscript_id,title,abstract,keywords,language,topic_category,ground_truth_reviewer_ids
M001,"Evaluasi Usability dan Perancangan Ulang Antarmuka Aplikasi Mobile E-Government Menggunakan Metode Design Thinking","Penelitian ini bertujuan untuk mengevaluasi tingkat kebergunaan (usability) aplikasi layanan publik pemerintah daerah dan merancang ulang antarmuka pengguna (UI/UX). Metode Design Thinking diterapkan melalui tahapan empathize, define, ideate, prototype, dan testing. Pengujian usability dilakukan menggunakan System Usability Scale (SUS) dan Maze kepada 50 responden. Hasil pengujian menunjukkan peningkatan skor SUS dari 54 menjadi 82 setelah perancangan ulang.","UI/UX; Design Thinking; System Usability Scale; Mobile Application; E-Government",ID,UI/UX,"R01;R06"
M002,"User Experience Evaluation of Mobile Banking Applications Using Heuristic Evaluation and Cognitive Walkthrough","This study evaluates user experience barriers in Indonesian digital banking applications. We combine Nielsen's Heuristic Evaluation with Cognitive Walkthrough sessions involving 15 UI/UX experts and 30 end-users. The findings reveal critical usability issues in navigation hierarchy and transaction confirmation flows, leading to a proposed high-fidelity interaction prototype.","User Experience; Heuristic Evaluation; Cognitive Walkthrough; Mobile Banking; Interaction Design",EN,UI/UX,"R01;R06"
M003,"Sistem Rekomendasi Reviewer Jurnal Ilmiah Berdasarkan Kemiripan Semantik Abstrak Menggunakan Multilingual Sentence-BERT","Penugasan reviewer jurnal secara manual sering terkendala oleh perbedaan bahasa dan keterbatasan pencocokan kata kunci leksikal. Penelitian ini membangun sistem rekomendasi reviewer otomatis menggunakan model pre-trained Multilingual Sentence-BERT dan perhitungan Cosine Similarity untuk memetakan kemiripan topik antara naskah masuk dan histori publikasi reviewer lintas bahasa Indonesia dan Inggris.","Sistem Rekomendasi; Multilingual Sentence-BERT; Cosine Similarity; Reviewer Assignment; NLP",ID,NLP,"R02;R07"
M004,"Aspect-Based Sentiment Analysis on Indonesian E-Commerce Product Reviews Using IndoBERT and BiLSTM","Online customer reviews contain rich feedback across multiple product aspects such as price, shipping, and quality. This paper proposes a hybrid deep learning architecture combining pre-trained IndoBERT embeddings with Bidirectional Long Short-Term Memory (BiLSTM) for aspect-based sentiment analysis on Indonesian marketplace datasets, achieving an F1-score of 91.4%.","Sentiment Analysis; IndoBERT; BiLSTM; Natural Language Processing; Text Mining",EN,NLP,"R02;R07"
M005,"Sistem Pendukung Keputusan Pemilihan Pemasok Bahan Baku Ramah Lingkungan Menggunakan Metode AHP dan TOPSIS","Pemilihan pemasok berkelanjutan membutuhkan evaluasi multi-kriteria yang objektif. Penelitian ini mengembangkan Sistem Pendukung Keputusan (SPK) dengan mengintegrasikan metode Analytical Hierarchy Process (AHP) untuk pembobotan kriteria dan Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) untuk perangkingan kandidat pemasok terbaik pada industri manufaktur.","Sistem Pendukung Keputusan; AHP; TOPSIS; Pemilihan Pemasok; Multi-Criteria Decision Making",ID,SPK,"R03;R08"
M006,"A Hybrid Fuzzy Multi-Criteria Decision-Making Model for Scholarship Recipient Selection in Higher Education","Selecting scholarship recipients involves subjective and uncertain evaluation criteria. This study implements a hybrid Fuzzy SAW and VIKOR decision support framework to rank student applicants based on academic achievement, socioeconomic status, and extracurricular participation, providing transparent decision recommendations for university committees.","Decision Support System; Fuzzy SAW; VIKOR; MCDM; Scholarship Selection",EN,SPK,"R03;R08"
M007,"Deteksi Penyakit Daun Tanaman Padi Secara Real-Time Menggunakan Arsitektur YOLOv8 Berbasis Citra Digital","Serangan hama dan penyakit pada tanaman padi dapat menurunkan hasil panen secara drastis. Penelitian ini menerapkan algoritma deep learning YOLOv8 untuk mendeteksi dan mengklasifikasikan empat jenis penyakit daun padi melalui tangkapan kamera smartphone. Model dilatih menggunakan 4.500 citra dengan teknik augmentasi data dan menghasilkan nilai mAP@50 sebesar 94,2%.","Computer Vision; YOLOv8; Deteksi Objek; Penyakit Tanaman Padi; Deep Learning",ID,Computer Vision,"R04;R09"
M008,"Convolutional Neural Network with Attention Mechanism for Automated Chest X-Ray Pneumonia Classification","Early diagnosis of pneumonia from chest radiographs is critical for clinical treatment. This paper presents a modified ResNet-50 architecture integrated with a Convolutional Block Attention Module (CBAM) to highlight lesion regions in chest X-ray images. Experimental results on public medical datasets demonstrate superior sensitivity and specificity compared to standard CNN baselines.","Medical Image Analysis; CNN; Attention Mechanism; ResNet; Pneumonia Detection",EN,Computer Vision,"R04;R09"
M009,"Rancang Bangun Sistem Pemantauan Kualitas Air Tambak Udang Vaname Berbasis Internet of Things dan LoRaWAN","Kualitas air seperti suhu, pH, dan kadar oksigen terlarut sangat menentukan keberhasilan budidaya udang vaname. Penelitian ini merancang sistem monitoring real-time berbasis mikrokontroler ESP32 dan protokol komunikasi LoRaWAN untuk transmisi data jarak jauh tanpa koneksi seluler, dilengkapi dashboard web untuk peringatan dini anomali kualitas air.","Internet of Things; LoRaWAN; ESP32; Kualitas Air; Smart Aquaculture",ID,IoT,"R05;R10"
M010,"Edge Computing and MQTT Protocol Optimization for Smart Home Energy Management Systems","High latency and bandwidth consumption are major bottlenecks in cloud-centric IoT smart homes. This research proposes an edge-computing gateway utilizing Raspberry Pi and lightweight MQTT messaging to monitor and schedule household appliance power consumption locally, reducing cloud data traffic by 68%.","Internet of Things; Edge Computing; MQTT; Smart Home; Energy Management",EN,IoT,"R05;R10"
M011,"Klasifikasi Risiko Kredit Macet Nasabah Perbankan Menggunakan Algoritma Extreme Gradient Boosting (XGBoost) dan SMOTE","Ketidakseimbangan kelas pada dataset kredit perbankan sering menyebabkan model machine learning bias terhadap kelas mayoritas. Studi ini mengombinasikan teknik Synthetic Minority Over-sampling Technique (SMOTE) dengan algoritma ensemble XGBoost untuk memprediksi risiko gagal bayar debitur, menghasilkan peningkatan nilai AUC-ROC hingga 0,93.","Data Mining; XGBoost; SMOTE; Risiko Kredit; Machine Learning",ID,Data Mining,"R02;R08"
M012,"Customer Churn Prediction in Telecommunication Services Using Random Forest and SHAP Explainable AI","Retaining existing subscribers is vital in the competitive telecom industry. We develop a customer churn prediction pipeline using Random Forest and LightGBM, paired with SHapley Additive exPlanations (SHAP) to interpret key behavioral features driving customer attrition for marketing retention strategies.","Customer Churn; Explainable AI; SHAP; Random Forest; Predictive Analytics",EN,Data Mining,"R02;R08"
M013,"Deteksi Serangan Distributed Denial of Service (DDoS) pada Jaringan Software-Defined Networking Menggunakan Deep Autoencoder","Arsitektur Software-Defined Networking (SDN) rentan terhadap serangan DDoS yang membanjiri kontroler pusat. Penelitian ini mengusulkan sistem deteksi intrusi (IDS) berbasis anomali menggunakan Deep Autoencoder dan Long Short-Term Memory (LSTM) untuk menganalisis trafik paket jaringan pada dataset CICDDoS2019 dengan akurasi deteksi mencapai 98,7%.","Keamanan Jaringan; DDoS; Software-Defined Networking; Intrusion Detection System; Deep Learning",ID,Cyber Security,"R05;R09"
M014,"Blockchain-Based Electronic Health Record Sharing Framework with Zero-Knowledge Proof Privacy Preservation","Sharing patient Electronic Health Records (EHR) across healthcare institutions poses severe data privacy and tampering risks. This study designs a permissioned Hyperledger Fabric blockchain architecture combined with Zero-Knowledge Proofs (ZKP) to authenticate medical access permissions without exposing sensitive patient identities.","Blockchain; Electronic Health Record; Zero-Knowledge Proof; Cyber Security; Data Privacy",EN,Cyber Security,"R05;R09"
M015,"Analisis Faktor Penerimaan Pengguna Terhadap Sistem Informasi Akademik Kampus Menggunakan Unified Theory of Acceptance and Use of Technology (UTAUT)","Keberhasilan implementasi sistem informasi akademik sangat bergantung pada penerimaan pengguna akhir. Penelitian ini menganalisis pengaruh performance expectancy, effort expectancy, social influence, dan facilitating conditions terhadap minat perilaku mahasiswa menggunakan metode Partial Least Squares Structural Equation Modeling (PLS-SEM) terhadap 320 responden.","Sistem Informasi; UTAUT; PLS-SEM; Penerimaan Teknologi; Perguruan Tinggi",ID,Sistem Informasi,"R01;R03"
M016,"Evaluating Enterprise Resource Planning (ERP) Post-Implementation Success Using the DeLone and McLean Information Systems Success Model","Many manufacturing firms struggle to realize full operational benefits after deploying ERP systems. This empirical study applies the updated DeLone and McLean IS Success Model to examine how system quality, information quality, and service quality impact net organizational benefits across 120 Indonesian SMEs.","ERP; DeLone and McLean; Information Systems Success; SMEs; Enterprise System",EN,Sistem Informasi,"R01;R03"
M017,"Peringkasan Teks Otomatis Berita Bahasa Indonesia Menggunakan Metode Extractive Berbasis TextRank dan IndoBERT Embedding","Tingginya volume berita daring menuntut sistem peringkasan teks otomatis yang mampu menyajikan informasi inti secara cepat. Penelitian ini menggabungkan representasi vektor kalimat dari IndoBERT dengan algoritma graf TextRank untuk mengekstrak kalimat paling representatif pada dataset IndoSum, dan dievaluasi menggunakan metrik ROUGE-1 dan ROUGE-2.","Peringkasan Teks; TextRank; IndoBERT; ROUGE; Pemrosesan Bahasa Alami",ID,NLP,"R02;R07"
M018,"Cross-Lingual Question Answering System for Academic Regulations Using Retrieval-Augmented Generation (RAG) and Multilingual LLMs","International and local students often struggle to query university academic regulations written in Indonesian. We build a Retrieval-Augmented Generation (RAG) chatbot powered by multilingual dense passage retrieval and large language models, enabling accurate bilingual question answering grounded in official institutional documents.","Retrieval-Augmented Generation; Question Answering; Multilingual NLP; Large Language Models; Vector Database",EN,NLP,"R02;R07"
M019,"Perancangan Desain Interaksi Aplikasi Pembelajaran Bahasa Isyarat Indonesia (BISINDO) dengan Pendekatan User-Centered Design dan Gamifikasi","Kurangnya media pembelajaran interaktif menyulitkan masyarakat umum dalam mempelajari Bahasa Isyarat Indonesia. Penelitian ini merancang prototipe aplikasi mobile edukasi BISINDO menggunakan pendekatan User-Centered Design (UCD) yang dipadukan dengan elemen gamifikasi Octalysis untuk meningkatkan motivasi dan keterlibatan belajar pengguna.","User-Centered Design; Gamifikasi; BISINDO; Desain Interaksi; Aplikasi Edukasi",ID,UI/UX,"R01;R06"
M020,"Accessibility Audit and Inclusive UI Design Guidelines for Visually Impaired Users on E-Commerce Platforms","Web accessibility remains overlooked in many rapidly growing online marketplaces. This study conducts a comprehensive accessibility audit based on WCAG 2.1 standards combined with screen-reader usability testing with visually impaired participants, resulting in actionable inclusive UI design patterns.","Web Accessibility; WCAG 2.1; Inclusive Design; UI/UX; Visually Impaired",EN,UI/UX,"R01;R06"
M021,"Implementasi Metode MOORA dan Pembobotan Entropy pada Sistem Pendukung Keputusan Penentuan Lokasi Gudang Distribusi Logistik","Pemilihan lokasi gudang distribusi yang strategis dapat menekan biaya rantai pasok perusahaan logistik. Penelitian ini mengombinasikan metode objektif Entropy untuk menentukan bobot kriteria secara otomatis dari data historis, serta metode Multi-Objective Optimization on the Basis of Ratio Analysis (MOORA) untuk menentukan alternatif lokasi terbaik.","Sistem Pendukung Keputusan; MOORA; Pembobotan Entropy; Logistik; Lokasi Gudang",ID,SPK,"R03;R08"
M022,"Group Decision Support System Using Borda Count and PROMETHEE II for Regional Infrastructure Priority Planning","Public infrastructure planning involves multiple stakeholders with conflicting priorities. This paper develops a Group Decision Support System (GDSS) integrating PROMETHEE II for individual decision-maker ranking and the Borda Count method to aggregate group consensus for municipal road construction projects.","Group Decision Support System; PROMETHEE II; Borda Count; Infrastructure Planning; Multi-Stakeholder",EN,SPK,"R03;R08"
M023,"Pengenalan Ekspresi Wajah Siswa pada Pembelajaran Daring Menggunakan Vision Transformer (ViT) untuk Evaluasi Keterlibatan Belajar","Memantau tingkat fokus dan keterlibatan emosi siswa selama kelas daring sinkronus merupakan tantangan bagi pengajar. Penelitian ini mengimplementasikan arsitektur Vision Transformer (ViT) untuk mengklasifikasikan tujuh ekspresi wajah mikro siswa secara real-time melalui webcam, mencapai akurasi klasifikasi sebesar 89,6% pada dataset FER2013.","Vision Transformer; Pengenalan Ekspresi Wajah; Computer Vision; Pembelajaran Daring; Affective Computing",ID,Computer Vision,"R04;R09"
M024,"Semantic Segmentation of Satellite Imagery for Urban Land Use Mapping Using U-Net and MobileNetV3 Backbone","Accurate urban land cover mapping is essential for sustainable city planning. This research proposes a lightweight U-Net semantic segmentation model utilizing a MobileNetV3 encoder backbone to segment buildings, roads, vegetation, and water bodies from high-resolution multispectral satellite imagery.","Semantic Segmentation; U-Net; Satellite Imagery; Remote Sensing; Computer Vision",EN,Computer Vision,"R04;R09"
M025,"Optimasi Penjadwalan Pengujian Regresi Perangkat Lunak pada Pipeline Continuous Integration Menggunakan Algoritma Genetika","Eksekusi seluruh test suite pada pengembangan perangkat lunak Agile memakan waktu komputasi yang tinggi pada pipeline CI/CD. Penelitian ini mengusulkan teknik Test Case Prioritization menggunakan Algoritma Genetika untuk mengurutkan kasus uji berdasarkan riwayat kegagalan dan cakupan kode, sehingga mempercepat deteksi bug dengan nilai APFD 0,91.","Rekayasa Perangkat Lunak; Continuous Integration; Test Case Prioritization; Algoritma Genetika; Software Testing",ID,Software Engineering,"R06;R10"
M026,"Microservices Architecture Migration and Performance Benchmarking from Monolithic Legacy E-Learning Systems","Monolithic e-learning platforms often suffer from scalability bottlenecks during peak examination periods. This paper presents a domain-driven design (DDD) migration strategy to decompose a legacy learning management system into containerized Docker and Kubernetes microservices, reducing response latency by 54%.","Microservices; Domain-Driven Design; Kubernetes; Software Architecture; Performance Benchmarking",EN,Software Engineering,"R06;R10"
M027,"Segmentasi Pelanggan Ritel Berdasarkan Pola Transaksi Menggunakan Model RFM dan Algoritma Clustering DBSCAN","Strategi pemasaran yang seragam untuk seluruh pelanggan ritel sering kali tidak efektif. Penelitian ini menggabungkan ekstraksi fitur Recency, Frequency, dan Monetary (RFM) dengan algoritma klasterisasi berbasis kepadatan DBSCAN dan K-Means untuk mengelompokkan segmen pelanggan loyal, potensial, dan berisiko berpindah.","Segmentasi Pelanggan; RFM Analysis; DBSCAN; K-Means Clustering; Data Mining",ID,Data Mining,"R02;R08"
M028,"Time-Series Forecasting of Renewable Solar Power Generation Using Hybrid CNN-BiGRU Network","Intermittent weather conditions make solar photovoltaic power generation difficult to predict accurately. We propose a hybrid deep learning model combining 1D Convolutional Neural Networks for local feature extraction and Bidirectional Gated Recurrent Units (BiGRU) for temporal sequence modeling of meteorological sensor data.","Time-Series Forecasting; CNN-BiGRU; Solar Energy; Deep Learning; Predictive Modeling",EN,Data Mining,"R04;R10"
M029,"Audit Tata Kelola Teknologi Informasi Menggunakan Framework COBIT 2019 pada Layanan Perbankan Digital","Transformasi digital perbankan menuntut tata kelola dan manajemen risiko TI yang selaras dengan tujuan bisnis perusahaan. Penelitian ini melakukan audit kapabilitas tata kelola TI menggunakan kerangka kerja COBIT 2019 pada domain Align, Plan, and Organize (APO) serta Build, Acquire, and Implement (BAI), disertai rekomendasi perbaikan kesenjangan level kapabilitas.","Tata Kelola TI; COBIT 2019; Audit Sistem Informasi; Manajemen Risiko TI; Perbankan Digital",ID,Sistem Informasi,"R01;R03"
M030,"Deteksi Website Phishing Berbasis Analisis Fitur URL dan Struktur HTML Menggunakan Ensemble Stacking Classifier","Serangan siber melalui situs web phishing terus berevolusi untuk mencuri kredensial pengguna. Penelitian ini mengekstraksi 35 fitur leksikal URL dan struktur DOM HTML, kemudian membangun model klasifikasi Ensemble Stacking yang menggungkan Random Forest, Support Vector Machine, dan CatBoost dengan tingkat akurasi 97,8%.","Phishing Detection; Ensemble Stacking; Cyber Security; Machine Learning; Web Security",ID,Cyber Security,"R05;R09"
"""

DEFAULT_PUBLICATIONS_CSV = """publication_id,reviewer_id,reviewer_name,title,abstract,keywords,language,publication_year,topic_category
P001,R01,"Dr. Andi Pratama, M.Kom.","Perancangan Antarmuka Pengguna Aplikasi Layanan Kesehatan Masyarakat Menggunakan Metode User-Centered Design","Penelitian ini berfokus pada perancangan ulang antarmuka pengguna (UI/UX) aplikasi pendaftaran pasien rumah sakit daerah menggunakan pendekatan User-Centered Design (UCD). Evaluasi dilakukan menggunakan System Usability Scale (SUS) terhadap 40 pengguna dan menghasilkan skor 84 yang masuk dalam kategori sangat baik.","UI/UX; User-Centered Design; System Usability Scale; Mobile Health",ID,2023,UI/UX
P002,R01,"Dr. Andi Pratama, M.Kom.","Usability Testing and Heuristic Evaluation on Indonesian FinTech Mobile Applications","Digital payment adoption relies heavily on intuitive interaction design. This study conducts Nielsen's 10 Heuristic Evaluation and usability testing sessions on three popular Indonesian e-wallet applications, identifying major cognitive friction points in authentication and transfer workflows.","Usability Testing; Heuristic Evaluation; FinTech; User Experience; Mobile App",EN,2024,UI/UX
P003,R01,"Dr. Andi Pratama, M.Kom.","Evaluasi Kesuksesan Implementasi Sistem Informasi Manajemen Rumah Sakit Menggunakan Model DeLone dan McLean","Studi empiris ini menguji pengaruh kualitas sistem, kualitas informasi, dan kualitas layanan terhadap kepuasan pengguna serta manfaat bersih organisasi menggunakan model kesuksesan sistem informasi DeLone dan McLean yang dianalisis dengan metode PLS-SEM.","DeLone and McLean; Sistem Informasi; PLS-SEM; Evaluasi Sistem",ID,2022,Sistem Informasi
P004,R01,"Dr. Andi Pratama, M.Kom.","IT Governance Capability Assessment Using COBIT 2019 Framework in State-Owned Financial Institutions","Effective alignment between business goals and information technology requires structured governance. This paper evaluates enterprise IT governance maturity levels across APO and BAI domains using the COBIT 2019 framework and formulates strategic improvement roadmaps.","COBIT 2019; IT Governance; Information Systems Audit; Risk Management",EN,2025,Sistem Informasi
P005,R02,"Dr. Budi Santoso, S.Kom., M.T.","Analisis Sentimen Berbasis Aspek pada Ulasan Aplikasi Pemerintahan Menggunakan Fine-Tuning IndoBERT","Ulasan pengguna di Google Play Store mengandung opini terhadap berbagai aspek layanan aplikasi. Penelitian ini menerapkan fine-tuning model pre-trained IndoBERT untuk mengekstraksi aspek layanan dan mengklasifikasikan polaritas sentimen positif, netral, dan negatif dengan akurasi 90,8%.","Analisis Sentimen; IndoBERT; Pemrosesan Bahasa Alami; Text Mining",ID,2024,NLP
P006,R02,"Dr. Budi Santoso, S.Kom., M.T.","Semantic Textual Similarity for Bilingual Academic Abstract Matching Using Multilingual Transformer Models","Matching scientific documents across Indonesian and English requires cross-lingual semantic understanding. We evaluate pre-trained multilingual Sentence-BERT embeddings combined with cosine similarity to measure topical relevance across bilingual scholarly corpora.","Multilingual Sentence-BERT; Semantic Similarity; Cosine Similarity; Cross-Lingual NLP",EN,2025,NLP
P007,R02,"Dr. Budi Santoso, S.Kom., M.T.","Prediksi Churn Pelanggan Layanan Streaming Digital Menggunakan Algoritma LightGBM dan Teknik Resampling SMOTE","Ketidakseimbangan kelas antara pelanggan yang berhenti berlangganan dan pelanggan aktif menurunkan performa model klasifikasi. Penelitian ini menerapkan SMOTE dan algoritma ensemble LightGBM serta interpretasi fitur SHAP untuk memprediksi potensi churn pelanggan secara akurat.","Customer Churn; LightGBM; SMOTE; SHAP; Data Mining",ID,2023,Data Mining
P008,R02,"Dr. Budi Santoso, S.Kom., M.T.","Retail Customer Segmentation Using RFM Model and Density-Based Spatial Clustering (DBSCAN)","Understanding purchasing behavior is critical for targeted retail marketing. This study extracts Recency, Frequency, and Monetary (RFM) attributes from transaction logs and compares K-Means with DBSCAN clustering to identify high-value customer segments.","RFM Analysis; DBSCAN; K-Means; Customer Segmentation; Data Mining",EN,2022,Data Mining
P009,R03,"Dr. Citra Dewi, M.MSI.","Sistem Pendukung Keputusan Penilaian Kinerja Karyawan Terbaik Menggunakan Kombinasi Metode AHP dan TOPSIS","Penilaian kinerja karyawan membutuhkan pembobotan kriteria yang konsisten dan perangkingan yang objektif. Penelitian ini menggabungkan metode Analytical Hierarchy Process (AHP) untuk menentukan bobot kriteria dan metode TOPSIS untuk merangking alternatif karyawan terbaik.","Sistem Pendukung Keputusan; AHP; TOPSIS; Multi-Criteria Decision Making",ID,2023,SPK
P010,R03,"Dr. Citra Dewi, M.MSI.","Multi-Criteria Decision Support System for Sustainable Warehouse Location Selection Using Entropy-MOORA","Selecting optimal logistics distribution centers involves conflicting economic and environmental criteria. This study integrates objective Shannon Entropy weighting with the MOORA method to rank candidate warehouse locations transparently.","Decision Support System; MOORA; Entropy Weighting; Logistics; MCDM",EN,2024,SPK
P011,R03,"Dr. Citra Dewi, M.MSI.","Analisis Penerimaan Pengguna Terhadap E-Learning Kampus Menggunakan Model UTAUT dan Structural Equation Modeling","Penelitian ini menginvestigasi determinan adopsi sistem pembelajaran daring oleh mahasiswa menggunakan kerangka Unified Theory of Acceptance and Use of Technology (UTAUT). Pengolahan data survei dengan PLS-SEM membuktikan pengaruh signifikan performance expectancy terhadap niat penggunaan.","UTAUT; PLS-SEM; Penerimaan Teknologi; Sistem Informasi Akademik",ID,2023,Sistem Informasi
P012,R03,"Dr. Citra Dewi, M.MSI.","Post-Implementation Evaluation of Cloud ERP Systems in Manufacturing Enterprises","Transitioning to cloud-based Enterprise Resource Planning (ERP) transforms organizational workflows. This research evaluates critical success factors and user satisfaction in Indonesian manufacturing companies following ERP deployment.","Enterprise Resource Planning; ERP; Information Systems; Organizational Impact",EN,2025,Sistem Informasi
P013,R04,"Dr. Dedi Kurniawan, M.Cs.","Identifikasi Otomatis Hama dan Penyakit Tanaman Hortikultura Menggunakan Arsitektur YOLOv7 Berbasis Citra Daun","Deteksi dini penyakit daun sangat penting untuk mencegah gagal panen. Penelitian ini melatih model deteksi objek real-time YOLOv7 pada ribuan citra daun tanaman yang diambil di lapangan dan mencapai nilai mean Average Precision (mAP) sebesar 93,5%.","Computer Vision; YOLO; Deteksi Penyakit Tanaman; Citra Digital; Deep Learning",ID,2023,Computer Vision
P014,R04,"Dr. Dedi Kurniawan, M.Cs.","Deep Convolutional Neural Networks with Spatial Attention for Automated Lung Disease Classification in Chest X-Rays","Automated analysis of thoracic radiographs assists radiologists in screening pulmonary abnormalities. We propose a ResNet-based CNN architecture enhanced with spatial and channel attention mechanisms to classify pneumonia and tuberculosis from X-ray images.","Medical Imaging; CNN; Attention Mechanism; Chest X-Ray; Computer Vision",EN,2024,Computer Vision
P015,R04,"Dr. Dedi Kurniawan, M.Cs.","Segmentasi Semantik Citra Penginderaan Jauh untuk Pemetaan Tutupan Lahan Perkotaan Menggunakan Arsitektur U-Net","Pemetaan tutupan lahan dari citra satelit resolusi tinggi membutuhkan akurasi piksel yang presisi. Penelitian ini menerapkan arsitektur encoder-decoder U-Net dengan backbone MobileNet untuk mengklasifikasikan area bangunan, jalan, dan vegetasi hijau.","Semantic Segmentation; U-Net; Citra Satelit; Remote Sensing; Deep Learning",ID,2025,Computer Vision
P016,R04,"Dr. Dedi Kurniawan, M.Cs.","Hybrid 1D-CNN and Bidirectional GRU Network for Short-Term Photovoltaic Wind and Solar Power Forecasting","Accurate renewable energy forecasting mitigates grid instability. This paper combines 1D Convolutional Neural Networks with Bidirectional Gated Recurrent Units (BiGRU) to capture spatial-temporal dependencies in historical weather and solar irradiance time-series data.","Time-Series Forecasting; CNN-BiGRU; Renewable Energy; Deep Learning",EN,2024,Data Mining
P017,R05,"Dr. Eka Wijaya, S.T., M.T.","Perancangan Sistem Pemantauan Kualitas Air Akuakultur Berbasis IoT Menggunakan Transmisi Jarak Jauh LoRa","Keterbatasan jaringan internet di area tambak pesisir menyulitkan pemantauan parameter air secara kontinu. Penelitian ini mengimplementasikan sensor pH, suhu, dan kekeruhan berbasis mikrokontroler ESP32 dengan modul komunikasi nirkabel LoRa untuk pemantauan real-time.","Internet of Things; LoRaWAN; ESP32; Kualitas Air; Akuakultur",ID,2024,IoT
P018,R05,"Dr. Eka Wijaya, S.T., M.T.","Lightweight MQTT and Edge Computing Architecture for Automated Smart Building Energy Monitoring","Centralized cloud processing introduces latency in smart building automation. We design a local edge computing gateway using Raspberry Pi and MQTT broker protocol to aggregate smart meter telemetry and optimize HVAC energy consumption.","Edge Computing; MQTT; Internet of Things; Smart Home; Energy Efficiency",EN,2023,IoT
P019,R05,"Dr. Eka Wijaya, S.T., M.T.","Deteksi Anomali Trafik Serangan DDoS pada Lingkungan Software-Defined Network (SDN) Menggunakan Deep Learning","Serangan Distributed Denial of Service (DDoS) dapat melumpuhkan kontroler terpusat pada jaringan SDN. Penelitian ini membangun Intrusion Detection System (IDS) menggunakan kombinasi Autoencoder dan Recurrent Neural Network untuk mendeteksi trafik berbahaya secara akurat.","Cyber Security; DDoS; Software-Defined Networking; Intrusion Detection System",ID,2025,Cyber Security
P020,R05,"Dr. Eka Wijaya, S.T., M.T.","Permissioned Blockchain and Cryptographic Access Control for Secure Healthcare Data Exchange","Protecting patient privacy during cross-hospital data sharing requires decentralized security mechanisms. This paper implements a Hyperledger Fabric blockchain framework combined with cryptographic proof protocols to secure Electronic Health Records (EHR).","Blockchain; Electronic Health Record; Data Privacy; Cyber Security; Cryptography",EN,2024,Cyber Security
P021,R06,"Dr. Fajar Nugroho, M.Kom.","Implementasi Design Thinking dan Gamifikasi pada Perancangan Antarmuka Aplikasi Edukasi Interaktif Anak","Untuk meningkatkan keterlibatan belajar siswa, penelitian ini menerapkan metode Design Thinking yang dipadukan dengan kerangka gamifikasi Octalysis dalam merancang desain interaksi aplikasi pembelajaran mobile, serta diuji menggunakan kognitif walkthrough.","Design Thinking; Gamifikasi; UI/UX; Desain Interaksi; Aplikasi Edukasi",ID,2024,UI/UX
P022,R06,"Dr. Fajar Nugroho, M.Kom.","Web Accessibility Evaluation and Inclusive User Interface Redesign Based on WCAG 2.1 Guidelines","Digital inclusivity requires web platforms to be navigable by users with visual and motor impairments. This research evaluates public service portals against WCAG 2.1 compliance standards and validates inclusive UI components with screen-reader users.","Web Accessibility; WCAG 2.1; Inclusive Design; User Interface; Usability",EN,2023,UI/UX
P023,R06,"Dr. Fajar Nugroho, M.Kom.","Prioritisasi Kasus Uji Regresi Otomatis Menggunakan Algoritma Metaheuristik pada Lingkungan Continuous Integration","Pengujian regresi pada pengembangan perangkat lunak Agile memerlukan efisiensi waktu eksekusi. Penelitian ini mengusulkan optimasi Test Case Prioritization berbasis Algoritma Genetika untuk mendeteksi kesalahan perangkat lunak lebih awal pada pipeline CI/CD.","Software Engineering; Test Case Prioritization; Algoritma Genetika; Continuous Integration",ID,2025,Software Engineering
P024,R06,"Dr. Fajar Nugroho, M.Kom.","Domain-Driven Decomposition and Containerized Microservices Migration for High-Traffic Web Applications","Legacy monolithic systems struggle to scale under concurrent user loads. This paper demonstrates a structured migration to Docker and Kubernetes microservices using Domain-Driven Design (DDD), significantly improving throughput and fault isolation.","Microservices; Domain-Driven Design; Kubernetes; Software Architecture; Cloud Native",EN,2024,Software Engineering
P025,R07,"Dr. Gita Permata, S.Kom., M.Cs.","Peringkasan Dokumen Berita Bahasa Indonesia Secara Ekstraktif Menggunakan Algoritma Graf TextRank dan Word Embedding","Penelitian ini mengembangkan sistem peringkasan teks otomatis (automatic text summarization) pada artikel berita berbahasa Indonesia dengan mengintegrasikan representasi vektor semantik IndoBERT ke dalam matriks ketetanggaan algoritma TextRank, dievaluasi dengan metrik ROUGE.","Peringkasan Teks; TextRank; IndoBERT; ROUGE; Natural Language Processing",ID,2023,NLP
P026,R07,"Dr. Gita Permata, S.Kom., M.Cs.","Retrieval-Augmented Generation (RAG) and Dense Passage Retrieval for Bilingual University Information Chatbots","Large Language Models often hallucinate when answering institution-specific queries. We implement a Retrieval-Augmented Generation (RAG) pipeline utilizing multilingual vector embeddings to provide grounded question answering over Indonesian and English academic documents.","Retrieval-Augmented Generation; Question Answering; Large Language Models; Multilingual NLP",EN,2025,NLP
P027,R07,"Dr. Gita Permata, S.Kom., M.Cs.","Klasifikasi Emosi dan Ujaran Kebencian pada Media Sosial Berbahasa Indonesia Menggunakan Arsitektur IndoBERT-BiLSTM","Teks media sosial di Indonesia memiliki karakteristik tidak baku dan multibahasa. Studi ini menggabungkan kontekstual embedding dari IndoBERT dengan jaringan saraf Bidirectional LSTM untuk mendeteksi ujaran kebencian dan sentimen dengan F1-score 89,7%.","IndoBERT; BiLSTM; Analisis Sentimen; Klasifikasi Teks; NLP",ID,2024,NLP
P028,R07,"Dr. Gita Permata, S.Kom., M.Cs.","Content-Based Scientific Paper and Reviewer Matching Using Pre-Trained Sentence Transformers","Automating editorial peer-review assignment requires accurate topical representation of scholarly texts. This paper evaluates Sentence-BERT models and cosine similarity scoring strategies to match submitted manuscripts with reviewer publication histories.","Sentence-BERT; Reviewer Assignment; Recommender System; Cosine Similarity; Text Mining",EN,2024,NLP
P029,R08,"Dr. Hendra Gunawan, M.Kom.","Sistem Pendukung Keputusan Kelompok Penentuan Prioritas Pembangunan Daerah Menggunakan Metode PROMETHEE II dan Borda","Pengambilan keputusan yang melibatkan banyak pemangku kepentingan membutuhkan metode agregasi konsensus. Penelitian ini membangun Group Decision Support System (GDSS) dengan metode PROMETHEE II dan fungsi pemungutan suara Borda Count.","Group Decision Support System; PROMETHEE; Borda Count; SPK Kelompok",ID,2024,SPK
P030,R08,"Dr. Hendra Gunawan, M.Kom.","Fuzzy Multi-Criteria Decision Making Using Fuzzy SAW and VIKOR for Academic Grant Allocation","Evaluating grant and scholarship proposals involves linguistic uncertainty in reviewer assessments. We propose a hybrid Fuzzy Simple Additive Weighting (SAW) and VIKOR model to generate robust ranking recommendations under uncertain criteria weights.","Fuzzy SAW; VIKOR; Decision Support System; MCDM; Higher Education",EN,2023,SPK
P031,R08,"Dr. Hendra Gunawan, M.Kom.","Prediksi Risiko Gagal Bayar Pembiayaan Kredit Menggunakan Ensemble Extreme Gradient Boosting (XGBoost)","Penilaian kelayakan kredit yang akurat dapat meminimalkan kredit macet (non-performing loan) di lembaga keuangan. Penelitian ini menerapkan algoritma ensemble XGBoost dan Random Forest dengan seleksi fitur untuk mengklasifikasikan profil risiko nasabah.","XGBoost; Random Forest; Risiko Kredit; Machine Learning; Data Mining",ID,2025,Data Mining
P032,R08,"Dr. Hendra Gunawan, M.Kom.","Explainable Machine Learning with SHAP for Customer Attrition and Credit Scoring Analytics","Black-box ensemble models often lack transparency required in financial and telecom decision-making. This study integrates SHapley Additive exPlanations (SHAP) with gradient boosted trees to provide interpretable feature attribution in predictive modeling.","Explainable AI; SHAP; Gradient Boosting; Predictive Analytics; Data Mining",EN,2024,Data Mining
P033,R09,"Dr. Indah Lestari, S.T., M.Eng.","Klasifikasi Ekspresi Wajah Mikro Secara Real-Time Menggunakan Arsitektur Vision Transformer (ViT) untuk Analisis Emosi","Pengenalan ekspresi wajah berbasis deep learning berperan penting dalam interaksi manusia dan komputer. Penelitian ini mengadaptasi model Vision Transformer (ViT) untuk mengenali tujuh kategori emosi wajah mahasiswa pada lingkungan pembelajaran daring.","Vision Transformer; Pengenalan Ekspresi Wajah; Computer Vision; Affective Computing",ID,2025,Computer Vision
P034,R09,"Dr. Indah Lestari, S.T., M.Eng.","Real-Time Agricultural Crop Disease Detection on Edge Devices Using Lightweight YOLOv8 Architecture","Deploying deep learning models for precision agriculture requires computational efficiency. This study optimizes YOLOv8 object detection to identify foliar plant diseases on mobile devices with high inference speed and mAP accuracy.","YOLOv8; Object Detection; Plant Disease; Computer Vision; Deep Learning",EN,2024,Computer Vision
P035,R09,"Dr. Indah Lestari, S.T., M.Eng.","Deteksi Situs Web Phishing Berdasarkan Ekstraksi Fitur Leksikal URL dan Struktur DOM Menggunakan Ensemble Learning","Kejahatan phishing di internet terus meningkat dengan memalsukan halaman login perbankan. Penelitian ini menggabungkan algoritma Random Forest, SVM, dan CatBoost dalam arsitektur Stacking Ensemble untuk mendeteksi URL phishing secara otomatis.","Phishing Detection; Ensemble Stacking; Cyber Security; Machine Learning; Web Security",ID,2023,Cyber Security
P036,R09,"Dr. Indah Lestari, S.T., M.Eng.","Anomaly-Based Network Intrusion Detection System Using Deep Autoencoders and LSTM Networks","Modern cyber attacks bypass traditional signature-based firewalls. We propose a hybrid deep learning Intrusion Detection System (IDS) combining Autoencoder feature compression with LSTM temporal analysis to detect zero-day network intrusions and DDoS traffic.","Intrusion Detection System; Autoencoder; LSTM; Network Security; Cyber Security",EN,2024,Cyber Security
P037,R10,"Dr. Joko Susilo, M.T.","Sistem Cerdas Pemantauan dan Kontrol Otomatis Kualitas Air Budidaya Perikanan Berbasis IoT dan Cloud","Penelitian ini merancang purwarupa sistem Internet of Things (IoT) terintegrasi sensor oksigen terlarut, suhu, dan pH air untuk mengotomatisasi aerator tambak udang serta menyajikan visualisasi data telemetri secara real-time melalui aplikasi web.","Internet of Things; Smart Aquaculture; Sensor Kualitas Air; Mikrokontroler",ID,2023,IoT
P038,R10,"Dr. Joko Susilo, M.T.","Performance Evaluation of Containerized Microservices and CI/CD Automation in Cloud-Based Learning Platforms","Continuous software delivery and elastic scaling are vital for modern educational platforms. This paper benchmarks Kubernetes microservices orchestration and automated regression testing pipelines against legacy monolithic deployments.","Microservices; CI/CD; Kubernetes; Software Engineering; Performance Benchmarking",EN,2024,Software Engineering
P039,R10,"Dr. Joko Susilo, M.T.","Peramalan Deret Waktu Beban Konsumsi Listrik dan Energi Surya Menggunakan Jaringan Saraf Tiruan Hibrida CNN-GRU","Fluktuasi cuaca mempengaruhi stabilitas produksi energi terbarukan panel surya. Penelitian ini membangun model prediksi time-series hibrida 1D-CNN dan Gated Recurrent Unit (GRU) untuk meramalkan output daya listrik harian dengan nilai RMSE yang rendah.","Time-Series Forecasting; CNN-GRU; Energi Surya; Deep Learning; IoT",ID,2025,Data Mining
P040,R10,"Dr. Joko Susilo, M.T.","Energy-Efficient Routing and LoRaWAN Network Optimization for Large-Scale Agricultural IoT Sensor Nodes","Battery lifetime is a primary constraint in wireless sensor networks deployed across rural farmlands. This study optimizes LoRaWAN spreading factors and MQTT payload compression to minimize power consumption in agricultural IoT nodes.","LoRaWAN; Wireless Sensor Network; Internet of Things; Energy Optimization; MQTT",EN,2024,IoT
"""


def ensure_csv_files_exist():
    if not os.path.exists(MS_PATH):
        with open(MS_PATH, "w", encoding="utf-8", newline="") as f:
            f.write(DEFAULT_MANUSCRIPTS_CSV.strip() + "\n")
    if not os.path.exists(PUB_PATH):
        with open(PUB_PATH, "w", encoding="utf-8", newline="") as f:
            f.write(DEFAULT_PUBLICATIONS_CSV.strip() + "\n")


# =====================================================================
# FUNGSI LOAD DATA & MODEL (DENGAN CACHE AGAR CEPAT)
# =====================================================================
@st.cache_data
def load_data():
    ensure_csv_files_exist()
    df_ms = pd.read_csv(MS_PATH)
    df_pub = pd.read_csv(PUB_PATH)

    df_ms["full_text"] = (
        df_ms["title"].fillna("") + ". " + df_ms["abstract"].fillna("")
    )
    df_pub["full_text"] = (
        df_pub["title"].fillna("") + ". " + df_pub["abstract"].fillna("")
    )
    df_ms["ground_truth_list"] = df_ms["ground_truth_reviewer_ids"].apply(
        lambda x: [r.strip() for r in str(x).split(";") if r.strip()]
    )
    return df_ms, df_pub


@st.cache_resource
def load_sbert_model(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)


@st.cache_data
def encode_corpus(_model, texts):
    return _model.encode(list(texts), convert_to_numpy=True)


# =====================================================================
# FUNGSI PERHITUNGAN KEMIRIPAN & PERANGKINGAN
# =====================================================================
def compute_query_similarity(
    query_text, df_pub, method, sbert_model, pub_sbert_vecs
):
    pub_texts = df_pub["full_text"].tolist()

    if method == "Multilingual Sentence-BERT":
        query_vec = sbert_model.encode([query_text], convert_to_numpy=True)
        scores = cosine_similarity(query_vec, pub_sbert_vecs)[0]
    elif method == "TF-IDF (Lexical Baseline)":
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        vectorizer.fit(pub_texts + [query_text])
        q_vec = vectorizer.transform([query_text])
        p_vecs = vectorizer.transform(pub_texts)
        scores = cosine_similarity(q_vec, p_vecs)[0]
    elif method == "BM25 (Lexical Baseline)":
        tokenized_pubs = [doc.lower().split() for doc in pub_texts]
        bm25 = BM25Okapi(tokenized_pubs)
        raw_scores = bm25.get_scores(query_text.lower().split())
        scores = (
            raw_scores / np.max(raw_scores)
            if np.max(raw_scores) > 0
            else raw_scores
        )
    return scores


def aggregate_and_rank(pub_scores, df_pub, strategy="Max Score (Max Pooling)"):
    df_temp = df_pub.copy()
    df_temp["sim_score"] = pub_scores

    reviewer_rows = []
    for rev_id, group in df_temp.groupby("reviewer_id"):
        sorted_group = group.sort_values(by="sim_score", ascending=False)
        scores = sorted_group["sim_score"].values
        best_row = sorted_group.iloc[0]

        if strategy == "Max Score (Max Pooling)":
            agg_score = float(np.max(scores))
        elif strategy == "Mean Score (Rata-rata Semua Artikel)":
            agg_score = float(np.mean(scores))
        else:
            agg_score = float(np.mean(scores[: min(3, len(scores))]))

        topics = ", ".join(group["topic_category"].unique())
        reviewer_rows.append(
            {
                "Reviewer ID": rev_id,
                "Nama Reviewer": best_row["reviewer_name"],
                "Bidang Kepakaran": topics,
                "Skor Kemiripan": agg_score,
                "Skor Maks": float(np.max(scores)),
                "Skor Rata-rata": float(np.mean(scores)),
                "Artikel Terdekat (Bukti Kepakaran)": best_row["title"],
                "Bahasa Artikel": best_row["language"],
                "Tahun": best_row["publication_year"],
            }
        )

    df_ranked = pd.DataFrame(reviewer_rows).sort_values(
        by="Skor Kemiripan", ascending=False
    )
    df_ranked.insert(0, "Peringkat", range(1, len(df_ranked) + 1))
    return df_ranked, df_temp


# =====================================================================
# FUNGSI EVALUASI KESELURUHAN DATASET
# =====================================================================
def evaluate_all(df_ms, df_pub, sbert_model, pub_sbert_vecs):
    ms_texts = df_ms["full_text"].tolist()
    pub_texts = df_pub["full_text"].tolist()

    ms_sbert_vecs = sbert_model.encode(ms_texts, convert_to_numpy=True)
    sim_msbert = cosine_similarity(ms_sbert_vecs, pub_sbert_vecs)

    vec_tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit(
        ms_texts + pub_texts
    )
    sim_tfidf = cosine_similarity(
        vec_tfidf.transform(ms_texts), vec_tfidf.transform(pub_texts)
    )

    bm25 = BM25Okapi([doc.lower().split() for doc in pub_texts])
    sim_bm25 = np.zeros((len(ms_texts), len(pub_texts)))
    for i, q in enumerate(ms_texts):
        sc = bm25.get_scores(q.lower().split())
        sim_bm25[i] = sc / np.max(sc) if np.max(sc) > 0 else sc

    configs = [
        ("TF-IDF (Max)", sim_tfidf, "Max Score (Max Pooling)"),
        ("BM25 (Max)", sim_bm25, "Max Score (Max Pooling)"),
        (
            "Multilingual SBERT (Mean)",
            sim_msbert,
            "Mean Score (Rata-rata Semua Artikel)",
        ),
        ("Multilingual SBERT (Top-3 Mean)", sim_msbert, "Top-3 Mean Score"),
        (
            "Multilingual SBERT (Max - Proposed)",
            sim_msbert,
            "Max Score (Max Pooling)",
        ),
    ]

    summary = []
    ndcg_per_method = {}

    for name, mat, strat in configs:
        p3_list, p5_list, r3_list, r5_list, mrr_list, ndcg5_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i, row in df_ms.iterrows():
            gt = row["ground_truth_list"]
            df_rk, _ = aggregate_and_rank(mat[i], df_pub, strategy=strat)
            recs = df_rk["Reviewer ID"].tolist()

            p3_list.append(len(set(recs[:3]).intersection(gt)) / 3.0)
            p5_list.append(len(set(recs[:5]).intersection(gt)) / 5.0)
            r3_list.append(len(set(recs[:3]).intersection(gt)) / len(gt))
            r5_list.append(len(set(recs[:5]).intersection(gt)) / len(gt))

            rr = 0.0
            for idx, r in enumerate(recs):
                if r in gt:
                    rr = 1.0 / (idx + 1)
                    break
            mrr_list.append(rr)

            dcg = sum(
                [
                    1.0 / np.log2(idx + 2)
                    for idx, r in enumerate(recs[:5])
                    if r in gt
                ]
            )
            idcg = sum(
                [1.0 / np.log2(idx + 2) for idx in range(min(len(gt), 5))]
            )
            ndcg5_list.append(dcg / idcg if idcg > 0 else 0.0)

        ndcg_per_method[name] = np.array(ndcg5_list)
        summary.append(
            {
                "Metode & Strategi Agregasi": name,
                "P@3": np.mean(p3_list),
                "P@5": np.mean(p5_list),
                "R@3": np.mean(r3_list),
                "R@5": np.mean(r5_list),
                "NDCG@5": np.mean(ndcg5_list),
                "MRR": np.mean(mrr_list),
            }
        )

    return pd.DataFrame(summary), ndcg_per_method


# =====================================================================
# UI UTAMA STREAMLIT
# =====================================================================
st.title("Sistem Rekomendasi Reviewer Jurnal Ilmiah")
st.markdown(
    "**Berdasarkan Kemiripan Topik Menggunakan Multilingual Sentence-BERT & Cosine Similarity**"
)

df_ms, df_pub = load_data()
sbert_model = load_sbert_model()
pub_sbert_vecs = encode_corpus(sbert_model, df_pub["full_text"])

st.sidebar.header("Parameter Sistem (SPK)")
selected_method = st.sidebar.selectbox(
    "Pilih Model Representasi Teks:",
    [
        "Multilingual Sentence-BERT",
        "TF-IDF (Lexical Baseline)",
        "BM25 (Lexical Baseline)",
    ],
)
selected_strategy = st.sidebar.selectbox(
    "Strategi Agregasi Profil Reviewer:",
    [
        "Max Score (Max Pooling)",
        "Mean Score (Rata-rata Semua Artikel)",
        "Top-3 Mean Score",
    ],
)
top_k = st.sidebar.slider(
    "Jumlah Rekomendasi (Top-K):", min_value=1, max_value=10, value=5
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Info Dataset:**\n"
    f"- Naskah Submisi: **{len(df_ms)} artikel** (ID & EN)\n"
    f"- Kandidat Reviewer: **{df_pub['reviewer_id'].nunique()} pakar**\n"
    f"- Histori Publikasi: **{len(df_pub)} artikel** (ID & EN)"
)

tab1, tab2, tab3 = st.tabs(
    [
        "1. Cari Rekomendasi Reviewer (Demo SPK)",
        "2. Evaluasi Model & Uji Statistik",
        "3. Database Profil Reviewer",
    ]
)

with tab1:
    st.subheader(
        "Simulasi Pencocokan Naskah Submisi Baru dengan Histori Reviewer"
    )
    input_mode = st.radio(
        "Sumber Input Naskah:",
        [
            "Pilih dari Dataset Uji (manuscripts.csv)",
            "Input Manual Naskah Baru (Ketik Judul & Abstrak)",
        ],
        horizontal=True,
    )

    if input_mode == "Pilih dari Dataset Uji (manuscripts.csv)":
        ms_options = (
            df_ms["manuscript_id"]
            + " ["
            + df_ms["language"]
            + " - "
            + df_ms["topic_category"]
            + "] : "
            + df_ms["title"]
        )
        selected_option = st.selectbox("Pilih Naskah Submisi:", ms_options)
        selected_idx = ms_options[ms_options == selected_option].index[0]
        row_ms = df_ms.iloc[selected_idx]

        input_title = st.text_input("Judul Naskah:", value=row_ms["title"])
        input_abstract = st.text_area(
            "Abstrak Naskah:", value=row_ms["abstract"], height=130
        )
        gt_info = f"**Ground Truth Reviewer (Data Historis/Pakar):** `{row_ms['ground_truth_reviewer_ids']}` | **Kategori Topik:** `{row_ms['topic_category']}`"
        st.markdown(gt_info)
    else:
        input_title = st.text_input(
            "Judul Naskah:",
            placeholder="Contoh: Perancangan UI/UX Aplikasi Mobile Kesehatan Menggunakan Design Thinking",
        )
        input_abstract = st.text_area(
            "Abstrak Naskah:",
            placeholder="Masukkan teks abstrak dalam Bahasa Indonesia atau Bahasa Inggris...",
            height=130,
        )

    if st.button("Proses & Rekomendasikan Reviewer", type="primary"):
        if not input_title.strip() and not input_abstract.strip():
            st.warning("Mohon isi Judul atau Abstrak naskah terlebih dahulu.")
        else:
            query_text = f"{input_title.strip()}. {input_abstract.strip()}"
            pub_scores = compute_query_similarity(
                query_text, df_pub, selected_method, sbert_model, pub_sbert_vecs
            )
            df_ranked, df_detail_scores = aggregate_and_rank(
                pub_scores, df_pub, strategy=selected_strategy
            )

            st.markdown(
                f"### Top-{top_k} Rekomendasi Reviewer ({selected_method} - {selected_strategy})"
            )
            st.dataframe(
                df_ranked.head(top_k).style.format(
                    {
                        "Skor Kemiripan": "{:.4f}",
                        "Skor Maks": "{:.4f}",
                        "Skor Rata-rata": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "#### Rincian Skor Kemiripan per Artikel pada Top-3 Reviewer"
            )
            for _, rev_row in df_ranked.head(3).iterrows():
                rev_id = rev_row["Reviewer ID"]
                with st.expander(
                    f"Peringkat {rev_row['Peringkat']}: {rev_row['Nama Reviewer']} ({rev_id}) - Skor: {rev_row['Skor Kemiripan']:.4f}"
                ):
                    sub_df = df_detail_scores[
                        df_detail_scores["reviewer_id"] == rev_id
                    ].sort_values(by="sim_score", ascending=False)
                    st.dataframe(
                        sub_df[
                            [
                                "publication_id",
                                "language",
                                "topic_category",
                                "publication_year",
                                "title",
                                "sim_score",
                            ]
                        ]
                        .rename(
                            columns={
                                "publication_id": "ID Pub",
                                "language": "Bahasa",
                                "topic_category": "Topik",
                                "publication_year": "Tahun",
                                "title": "Judul Publikasi Reviewer",
                                "sim_score": "Cosine Similarity",
                            }
                        )
                        .style.format({"Cosine Similarity": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )

with tab2:
    st.subheader(
        "Komparasi Performa Metode & Strategi Agregasi (Seluruh Naskah Uji)"
    )
    df_eval_summary, ndcg_dict = evaluate_all(
        df_ms, df_pub, sbert_model, pub_sbert_vecs
    )

    st.dataframe(
        df_eval_summary.style.format(
            {
                "P@3": "{:.4f}",
                "P@5": "{:.4f}",
                "R@3": "{:.4f}",
                "R@5": "{:.4f}",
                "NDCG@5": "{:.4f}",
                "MRR": "{:.4f}",
            }
        ).highlight_max(
            subset=["P@3", "P@5", "R@3", "R@5", "NDCG@5", "MRR"],
            color="#d4edda",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### Uji Signifikansi Statistik (Paired Wilcoxon Signed-Rank Test pada NDCG@5)"
    )
    prop_scores = ndcg_dict["Multilingual SBERT (Max - Proposed)"]
    wilcoxon_rows = []
    for base_name in [
        "TF-IDF (Max)",
        "BM25 (Max)",
        "Multilingual SBERT (Mean)",
    ]:
        base_scores = ndcg_dict[base_name]
        diff = prop_scores - base_scores
        if np.any(diff != 0):
            stat, p_val = wilcoxon(
                prop_scores, base_scores, alternative="greater"
            )
        else:
            stat, p_val = 0.0, 1.0
        wilcoxon_rows.append(
            {
                "Metode Utama": "Multilingual SBERT (Max)",
                "Metode Pembanding": base_name,
                "Selisih Rata-rata NDCG@5": f"+{np.mean(diff):.4f}",
                "p-value": f"{p_val:.6f}",
                "Kesimpulan (alpha = 0.05)": (
                    "Signifikan Lebih Unggul (p < 0.05)"
                    if p_val < 0.05
                    else "Tidak Signifikan"
                ),
            }
        )
    st.table(pd.DataFrame(wilcoxon_rows))

with tab3:
    st.subheader(
        "Daftar Kandidat Reviewer & Arsip Publikasi (Multilingual & Multi-Bidang)"
    )
    rev_list = [
        f"{r_id} - {r_name}"
        for r_id, r_name in zip(
            df_pub["reviewer_id"].unique(),
            df_pub.groupby("reviewer_id")["reviewer_name"].first().values,
        )
    ]
    selected_rev = st.selectbox(
        "Filter berdasarkan Reviewer:", ["Semua Reviewer"] + rev_list
    )
    if selected_rev == "Semua Reviewer":
        st.dataframe(df_pub, use_container_width=True, hide_index=True)
    else:
        rev_code = selected_rev.split(" - ")[0]
        st.dataframe(
            df_pub[df_pub["reviewer_id"] == rev_code],
            use_container_width=True,
            hide_index=True,
        )