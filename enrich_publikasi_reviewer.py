import requests
import pandas as pd
import time

def reconstruct_abstract(inverted_index):
    """Mengubah format abstract_inverted_index dari OpenAlex menjadi teks utuh."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join([w for _, w in word_positions])

def fetch_reviewer_publications_openalex(reviewer_id, clean_name, max_papers=10):
    """Mencari profil author di OpenAlex dan mengambil riwayat publikasinya."""
    search_url = "https://api.openalex.org/authors"
    params = {"search": clean_name, "per-page": 1}
    res = requests.get(search_url, params=params, timeout=15).json()
    
    results = res.get("results", [])
    if not results:
        print(f"   [!] Profil tidak ditemukan di OpenAlex untuk: {clean_name}")
        return [], ""

    author_data = results[0]
    author_id = author_data["id"]
    
    # Ambil Topic of Interest otomatis dari konsep/topik teratas milik author tersebut
    topics = [t["display_name"] for t in author_data.get("topics", [])[:5]]
    if not topics:
        topics = [c["display_name"] for c in author_data.get("x_concepts", [])[:5]]
    topic_of_interest = ", ".join(topics)

    # Ambil daftar publikasi (works) milik reviewer tersebut
    works_url = "https://api.openalex.org/works"
    works_params = {
        "filter": f"author.id:{author_id}",
        "sort": "cited_by_count:desc",
        "per-page": max_papers
    }
    works_res = requests.get(works_url, params=works_params, timeout=15).json()
    
    publications = []
    for idx, work in enumerate(works_res.get("results", []), 1):
        title = work.get("title") or ""
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        
        # Ambil nama jurnal / konferensi tempat publish
        primary_loc = work.get("primary_location") or {}
        source_info = primary_loc.get("source") or {}
        venue_name = source_info.get("display_name") or "Jurnal / Prosiding Ilmiah"
        pub_year = work.get("publication_year") or ""

        if title:
            publications.append({
                "ID_Publikasi": f"PUB-{reviewer_id.split('-')[-1]}-{idx:02d}",
                "ID_Reviewer": reviewer_id,
                "Nama_Reviewer": clean_name,
                "Judul_Publikasi": title,
                "Abstrak_Publikasi": abstract if abstract else "(Abstrak perlu dilengkapi manual dari Google Scholar/Garuda)",
                "Tahun_Publish": pub_year,
                "Tempat_Publish": f"{venue_name} ({pub_year})",
                "DOI_atau_Link": work.get("doi") or work.get("id")
            })
            
    return publications, topic_of_interest


if __name__ == "__main__":
    # Contoh daftar nama bersih (tanpa gelar) agar pencarian API akurat
    # Nanti sesuaikan dengan ke-12 nama dari file '1_dataset_reviewer_sibc.csv'
    daftar_reviewer = [
        ("REV-01", "Hazriani"),
        ("REV-02", "Budi Rahmadya"),
        # Tambahkan 10 nama reviewer SIBC lainnya di sini tanpa gelar (misal: ("REV-03", "Nama Lengkap"))
    ]

    all_pubs = []
    updated_topics = []

    for rev_id, clean_name in daftar_reviewer:
        print(f"Mencari histori publikasi untuk {rev_id} - {clean_name}...")
        pubs, topics = fetch_reviewer_publications_openalex(rev_id, clean_name, max_papers=10)
        all_pubs.extend(pubs)
        updated_topics.append({
            "ID_Reviewer": rev_id,
            "Nama_Bersih": clean_name,
            "Suggested_Topic_of_Interest": topics,
            "Jumlah_Publikasi_Ditemukan": len(pubs)
        })
        time.sleep(1)

    df_pubs = pd.DataFrame(all_pubs)
    df_pubs.to_csv("3_dataset_publikasi_reviewer.csv", index=False, encoding="utf-8-sig")
    
    df_topics = pd.DataFrame(updated_topics)
    df_topics.to_csv("ringkasan_topik_reviewer.csv", index=False, encoding="utf-8-sig")
    print("\nSelesai! Histori publikasi disimpan ke '3_dataset_publikasi_reviewer.csv'")