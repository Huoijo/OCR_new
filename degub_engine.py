from src.ocr_cpu.ocr.engine import create_ocr_engine

engine = create_ocr_engine(
    backend="tesseract",
    lang="vie+eng",
    psm=11,
)

print(engine.available_languages())

result = engine.recognize("data/raw/test_images/test_images/images/img_2934.jpg")

print("TEXT:")
print(result.text)

print("AVG CONF:", result.avg_conf)
print("N BOXES:", result.n_boxes)
print("LATENCY MS:", result.latency_ms)
print("ERROR:", result.error)