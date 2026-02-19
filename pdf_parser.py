"""
Parser robusto per PDF con estrazione metadata, testo e immagini.
Crea una struttura standardizzata con citazioni verificabili.
"""

import pdfplumber
import os
from pathlib import Path
from typing import Dict, List, Any
import json
import hashlib

class PDFMetadataExtractor:
    """Estrae metadati, testo e immagini dai PDF mantenendo traccia delle posizioni."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.filename = Path(pdf_path).name
        self.metadata = {
            "filename": self.filename,
            "total_pages": 0,
            "chunks": []
        }
    
    def extract(self) -> Dict[str, Any]:
        """Estrae tutto dal PDF con metadata."""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.metadata["total_pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    chunk = self._extract_page_chunk(page, page_num)
                    self.metadata["chunks"].append(chunk)
            
            return self.metadata
        except Exception as e:
            print(f"❌ Errore parsing {self.filename}: {e}")
            return None
    
    def _extract_page_chunk(self, page, page_num: int) -> Dict:
        """Estrae testo e metadati da una singola pagina."""
        chunk = {
            "page": page_num,
            "chunk_id": f"{Path(self.filename).stem}_p{page_num}",
            "text": page.extract_text() or "",
            "tables": [],
            "images_count": len(page.images) if page.images else 0,
            "bbox": page.bbox if hasattr(page, 'bbox') else None
        }
        
        # Estrai tabelle se presenti
        if page.tables:
            for table in page.tables:
                chunk["tables"].append({
                    "data": table,
                    "bbox": table.bbox if hasattr(table, 'bbox') else None
                })
        
        return chunk


def parse_all_pdfs(data_dir: str) -> List[Dict]:
    """Parsa tutti i PDF in una cartella."""
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  Nessun PDF trovato in {data_dir}")
        return []
    
    print(f"🔍 Trovati {len(pdf_files)} PDF. Parsing in corso...")
    
    all_metadata = []
    for pdf_file in sorted(pdf_files):
        print(f"  ↳ {pdf_file.name}...", end="")
        extractor = PDFMetadataExtractor(str(pdf_file))
        metadata = extractor.extract()
        if metadata:
            all_metadata.append(metadata)
            print(f" ✅ ({metadata['total_pages']} pagine)")
        else:
            print(" ❌")
    
    return all_metadata


def save_metadata(metadata_list: List[Dict], output_path: str):
    """Salva i metadati in JSON per reference futuro."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadati salvati in {output_path}")


if __name__ == "__main__":
    DATA_DIR = "./data"
    output = parse_all_pdfs(DATA_DIR)
    save_metadata(output, "./pdf_metadata.json")
