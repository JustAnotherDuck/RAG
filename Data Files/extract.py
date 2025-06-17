import time
import json
import traceback
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any
import shutil

# Configure logging)

class PDFExtractionTester:
    def __init__(self):
        self.base_dir = Path("extraction_results")
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure if it doesn't exist"""
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "logs").mkdir(exist_ok=True)
        
    def create_document_folders(self, pdf_name: str) -> Dict[str, Path]:
        """Create folder structure for a specific document"""
        doc_dir = self.base_dir / pdf_name
        doc_dir.mkdir(exist_ok=True)
        return {
            "text": doc_dir / "text_files",
        }
    
    def save_extracted_text(self, folder: Path, pdf_name: str, library: str, text: str) -> Path:
        """Save extracted text to file"""
        folder.mkdir(exist_ok=True)
        text_file = folder / f"{pdf_name}_{library}_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return text_file
    

    def test_unstructured(self, pdf_path: Path) -> Dict[str, Any]:
        """Test Unstructured library"""
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(pdf_path, strategy="auto")
        text = "\n\n".join([str(el) for el in elements])
        
        return {
            "text": text,
        }


    def test_pypdf2(self, pdf_path: Path) -> Dict[str, Any]:
        """Test PyPDF2 library"""
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        return {
            "text": text,
        }

    def test_tesseract(self, pdf_path: Path) -> Dict[str, Any]:
        """Test Tesseract OCR approach"""
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        
        return {
            "text": text,
        }

    def run_with_error_handling(self, func, pdf_path: Path, library_name: str) -> Dict[str, Any]:
        """Wrapper function with error handling"""
        result = {"library": library_name}
        start_time = time.time()
        
        try:
            extraction_result = func(pdf_path)
            result.update({
                "status": "Success",
                "time": time.time() - start_time,
                **extraction_result
            })
        except Exception as e:
            result.update({
                "status": "Failed",
                "time": time.time() - start_time,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        
        return result

    def run_tests(self, pdf_path: Path):
        """Run all tests on a given PDF with proper error handling"""
        if not pdf_path.exists():
            return None
            
        pdf_name = pdf_path.stem
        folders = self.create_document_folders(pdf_name)
                
        tests = {
            "Unstructured": self.test_unstructured,
            "PyPDF2": self.test_pypdf2,
            "Tesseract": self.test_tesseract
        }
        
        results = {}
        for lib_name, test_func in tests.items():
            result = self.run_with_error_handling(test_func, pdf_path, lib_name)
            
            if result["status"] == "Success":
                text_file = self.save_extracted_text(
                    folders["text"], pdf_name, lib_name, result["text"]
                )
                result["text_file"] = str(text_file)
                        
            results[lib_name] = result
        
        return results


if __name__ == "__main__":
    tester = PDFExtractionTester()
    
    # Example usage - modify this to point to your PDF
    pdf_path = Path("2407.13035v1.pdf")  # Change this to your PDF filename
    if pdf_path.exists():
        tester.run_tests(pdf_path)
