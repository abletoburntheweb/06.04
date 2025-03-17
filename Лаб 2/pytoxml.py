import xml.etree.ElementTree as ET

root = ET.Element("sofas")

sofas_data = [
    {"model": "Villaggio угловой малый пр\лев Basic", "brand": "Villaggio", "price": 167910},
    {"model": "Brownie 3-х местный Premier", "brand": "Brownie", "price": 200600},
    {"model": "Brioche FIRST 2-х местный с широкими подлокотниками", "brand": "Brioche", "price": 132360},
    {"model": "Rimini Basic", "brand": "Rimini", "price": 145000},
    {"model": "Monaco модульный FIRST левый без дополнений", "brand": "Monaco", "price": 180070},
    {"model": "Диван Brownie Memori 2-х местный Optimum", "brand": "Brownie", "price": 148020},
]

for sofa in sofas_data:
    sofa_element = ET.SubElement(root, "sofa")
    model_element = ET.SubElement(sofa_element, "model")
    model_element.text = sofa["model"]
    brand_element = ET.SubElement(sofa_element, "brand")
    brand_element.text = sofa["brand"]
    price_element = ET.SubElement(sofa_element, "price")
    price_element.text = str(sofa["price"])

tree = ET.ElementTree(root)
with open("sofas.xml", "wb") as file:
    tree.write(file, encoding='utf-8', xml_declaration=True)

print("XML файл успешно создан!")
