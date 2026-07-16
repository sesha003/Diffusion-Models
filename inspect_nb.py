import json
NB = r"c:\Users\OM\Downloads\unet model\unet model\Diffusion_Model\Code\Lung_Segmentation_Final.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    print(f"\n--- Cell {i} [{c['cell_type']}] ---")
    print(src[:200])
