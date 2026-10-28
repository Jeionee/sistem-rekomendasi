import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import urllib3

# Nonaktifkan warning SSL jika sertifikat kampus kadang bermasalah
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

BASE_URL = "https://sibc.upnjatim.ac.id/index.php/sibc"

# ==========================================
# BAGIAN 1: SCRAPE DAFTAR REVIEWER SIBC
# ==========================================
def scrape_reviewers():
    url = f"{BASE_URL}/reviewer"
    print(f"[1/2] Mengambil daftar reviewer dari: {url} ...")
    res = requests.get(url, headers=HEADERS, verify=False, timeout=20)
    soup = BeautifulSoup(res.text, "html.parser")
    
    # Cari konten utama halaman OJS
    main_content = soup.find("div", class_="page") or soup.find("main") or soup
    
    reviewers = []
    # Di OJS, daftar reviewer biasanya ditulis dalam tag <p> atau <li>
    items = main_content.find_all(["li", "p"])
    
    rev_id = 1
    for item in items:
        text = " ".join(item.get_text(" ", strip=True).split())
        # Filter baris yang mengandung gelar akademik khas daftar reviewer
        if any(gelar in text for gelar in ["Dr.", "M.Kom", "M.T", "M.Cs", "M.Eng", "Ph.D", "S.Kom", "Prof."]):
            # Hindari teks footer/editor in chief jika ikut terjaring
            if "Editor in Chief" in text or "Jl Raya Rungkut" in text:
                continue
                
            reviewers.append({
                "ID_Reviewer": f"REV-{rev_id:02d}",
                "Raw_Text_SIBC": text,
                "Nama_Reviewer": text.split("Universitas")[0].split("Institut")[0].split("Politeknik")[0].strip(" ,;-"),
                "Afiliasi_Reviewer": text,
                "Topic_of_Interest": "",      # Akan diperkaya di Script 2 / SINTA / Scholar
                "Expertise_History": "",      # Ringkasan histori kepakaran
                "Kontak_Email": ""            # Diisi untuk tampilan UI Top-5
            })
            rev_id += 1

    df_rev = pd.DataFrame(reviewers)
    df_rev.to_csv("1_dataset_reviewer_sibc.csv", index=False, encoding="utf-8-sig")
    print(f"    -> Berhasil menyimpan {len(df_rev)} reviewer ke '1_dataset_reviewer_sibc.csv'\n")
    return df_rev


# ==========================================
# BAGIAN 2: SCRAPE ARSIP ARTIKEL SIBC
# ==========================================
def get_issue_links(max_pages=2):
    issue_links = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/issue/archive" if page == 1 else f"{BASE_URL}/issue/archive/{page}"
        res = requests.get(url, headers=HEADERS, verify=False, timeout=20)
        if res.status_code != 200:
            break
        soup = BeautifulSoup(res.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/issue/view/" in href and href not in issue_links:
                issue_links.append(href)
    return issue_links

def get_article_links_from_issue(issue_url):
    res = requests.get(issue_url, headers=HEADERS, verify=False, timeout=20)
    soup = BeautifulSoup(res.text, "html.parser")
    article_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Pastikan hanya link halaman detail artikel, bukan link download file PDF langsung
        if "/article/view/" in href:
            parts = href.rstrip("/").split("/")
            if parts[-2] == "view":  # Format standar: .../article/view/{id}
                if href not in article_links:
                    article_links.append(href)
    return article_links

def parse_article_detail(article_url):
    res = requests.get(article_url, headers=HEADERS, verify=False, timeout=20)
    soup = BeautifulSoup(res.text, "html.parser")
    
    # Ekstrak ID Artikel dari URL
    art_id_num = article_url.rstrip("/").split("/")[-1]
    
    # 1. Judul (Dari meta citation_title atau DC.Title)
    title_meta = soup.find("meta", {"name": "citation_title"}) or soup.find("meta", {"name": "DC.Title"})
    judul = title_meta["content"].strip() if title_meta and title_meta.get("content") else ""
    if not judul:
        h1 = soup.find("h1", class_="page_title")
        judul = h1.get_text(strip=True) if h1 else ""

    # 2. Abstrak (Dari meta DC.Description atau section abstract)
    abs_meta = soup.find("meta", {"name": "DC.Description"})
    abstrak = abs_meta["content"].strip() if abs_meta and abs_meta.get("content") else ""
    if not abstrak:
        abs_sec = soup.find("section", class_="item abstract") or soup.find("div", class_="abstract")
        if abs_sec:
            abstrak = abs_sec.get_text(" ", strip=True).replace("Abstract", "").strip()

    # 3. Keyword (Dari meta citation_keywords atau DC.Subject)
    kw_metas = soup.find_all("meta", {"name": re.compile(r"citation_keywords|DC\.Subject", re.I)})
    keywords = [m["content"].strip() for m in kw_metas if m.get("content")]
    if not keywords:
        kw_sec = soup.find("section", class_="item keywords")
        if kw_sec:
            keywords = [ kw_sec.get_text(" ", strip=True).replace("Keywords:", "").strip() ]
    keyword_str = ", ".join(dict.fromkeys(keywords))

    # 4. Nama Author & Afiliasi
    author_metas = soup.find_all("meta", {"name": "citation_author"})
    authors = [m["content"].strip() for m in author_metas if m.get("content")]
    
    affil_metas = soup.find_all("meta", {"name": "citation_author_institution"})
    affils = [m["content"].strip() for m in affil_metas if m.get("content")]

    # 5. Tahun / Tanggal Terbit & Edisi
    date_meta = soup.find("meta", {"name": "citation_date"})
    tanggal = date_meta["content"].strip() if date_meta and date_meta.get("content") else ""

    # Abaikan item non-artikel seperti "Front Cover" atau "Preface"
    if any(skip in judul.lower() for skip in ["front cover", "preface", "daftar isi", "table of content"]):
        return None

    return {
        "ID_Artikel": f"ART-{art_id_num}",
        "Judul": judul,
        "Abstrak": " ".join(abstrak.split()),
        "Keyword": keyword_str,
        "Nama_Author": "; ".join(authors),
        "Afiliasi_Author": "; ".join(dict.fromkeys(affils)),
        "Tanggal_Terbit": tanggal,
        "URL_Artikel": article_url
    }

def scrape_articles(max_issues=6):
    print(f"[2/2] Mengambil daftar edisi (issue) dari arsip SIBC...")
    issue_links = get_issue_links(max_pages=2)[:max_issues]
    print(f"    -> Ditemukan {len(issue_links)} edisi yang akan di-scrape.")

    all_articles = []
    for idx, issue_url in enumerate(issue_links, 1):
        print(f"    -> [{idx}/{len(issue_links)}] Membuka edisi: {issue_url}")
        art_links = get_article_links_from_issue(issue_url)
        for art_url in art_links:
            try:
                data = parse_article_detail(art_url)
                if data and data["Abstrak"]:
                    all_articles.append(data)
                    print(f"       + Berhasil ambil: {data['ID_Artikel']} | {data['Judul'][:50]}...")
                time.sleep(0.8)  # Jeda sopan agar server kampus tidak overload
            except Exception as e:
                print(f"       ! Gagal pada {art_url}: {e}")

    df_art = pd.DataFrame(all_articles)
    df_art.to_csv("2_dataset_naskah_artikel_sibc.csv", index=False, encoding="utf-8-sig")
    print(f"\nSelesai! Total {len(df_art)} artikel berhasil disimpan ke '2_dataset_naskah_artikel_sibc.csv'")
    return df_art

if __name__ == "__main__":
    df_reviewer = scrape_reviewers()
    # Ubah max_issues=10 jika ingin mengambil lebih banyak edisi tahun-tahun sebelumnya
    df_artikel = scrape_articles(max_issues=6)