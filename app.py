from flask import Flask, request, render_template_string, send_file
import tempfile, os
import cv2
import numpy as np
from ortools.sat.python import cp_model

app = Flask(__name__)

# Region detection as before
from typing import List, Tuple, Dict

def detect_regions_from_image(img: np.ndarray, grid_size: Tuple[int,int]) -> List[List[int]]:
    rows, cols = grid_size
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thick = cv2.morphologyEx(bw, cv2.MORPH_OPEN, thin_kernel)
    interior = cv2.bitwise_not(thick)
    num_labels, labels = cv2.connectedComponents(interior)
    cell_h = h/rows; cell_w = w/cols
    region_map = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            cy = int((i+0.5)*cell_h); cx = int((j+0.5)*cell_w)
            lbl = labels[cy, cx]
            region_map[i,j] = lbl
    unique = sorted(set(region_map.flatten()) - {0})
    mapping = {old: new for new, old in enumerate(unique, start=1)}
    for i in range(rows):
        for j in range(cols):
            region_map[i,j] = mapping.get(region_map[i,j], 0)
    return region_map.tolist()

# Solver function

def solve_star_battle(regions: List[List[int]], stars_per: int,
                      time_limit_s: int = 10, workers: int = 4):
    rows = len(regions); cols = len(regions[0])
    model = cp_model.CpModel()
    x = {}
    for r in range(rows):
        for c in range(cols):
            x[(r,c)] = model.NewBoolVar(f'x_{r}_{c}')
    for r in range(rows): model.Add(sum(x[(r,c)] for c in range(cols)) == stars_per)
    for c in range(cols): model.Add(sum(x[(r,c)] for r in range(rows)) == stars_per)
    region_cells: Dict[int, List[Tuple[int,int]]] = {}
    for r in range(rows):
        for c in range(cols):
            rid = regions[r][c]
            region_cells.setdefault(rid, []).append((r,c))
    for cells in region_cells.values(): model.Add(sum(x[pos] for pos in cells) == stars_per)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for r in range(rows):
        for c in range(cols):
            for dr, dc in neighbors:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols:
                    model.Add(x[(r,c)]+x[(nr,nc)] <= 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = workers
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [(r,c) for r in range(rows) for c in range(cols) if solver.Value(x[(r,c)])]
    return []

# HTML template
TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Star Battle Screenshot Solver 🚀</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #eef2f7; margin: 0; padding: 0; }
    .container { max-width: 900px; margin: 3em auto; background: #fff; padding: 2.5em; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-radius: 10px; }
    h1 { text-align: center; margin-bottom: 0.2em; font-size: 2.5em; color: #333; }
    p.tagline { text-align: center; margin-top: 0; margin-bottom: 1.5em; color: #666; font-style: italic; }
    form { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2em; align-items: center; }
    form input[type="file"] { grid-column: 1 / span 2; }
    form label { font-weight: 600; color: #444; }
    form input[type="number"] { padding: 0.6em; font-size: 1em; border: 1px solid #ccc; border-radius: 4px; }
    form input[type="submit"] { padding: 0.8em; font-size: 1.1em; background: #007bff; color: #fff; border: none; border-radius: 6px; cursor: pointer; transition: background 0.2s; grid-column: 1 / span 2; }
    form input[type="submit"]:hover { background: #0056b3; }
    .result { margin-top: 2.5em; text-align: center; }
    .result h2 { color: #333; }
    .result img { max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 6px; margin-top: 1em; }
    footer { text-align: center; margin-top: 3em; color: #aaa; font-size: 0.9em; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Star Battle Screenshot Solver</h1>
    <p class="tagline">Upload your puzzle screenshot, and let the stars align! 🌟</p>
    <form method="post" enctype="multipart/form-data">
      <label for="file">Puzzle Screenshot:</label>
      <input type="file" id="file" name="file" accept="image/*" required>
      <label for="stars">Stars per row/col/region:</label>
      <input type="number" id="stars" name="stars" value="2" min="1" required>
      <label for="size">Grid size (N×N):</label>
      <input type="number" id="size" name="size" value="10" min="1" required>
      <input type="submit" value="💫 Solve My Puzzle!">
    </form>
    {% if output_img %}
    <div class="result">
      <h2>Your Puzzle, Solved</h2>
      <img src="/result/{{output_img}}" alt="Solved Puzzle">
    </div>
    {% endif %}
    <footer>Powered by Python, OpenCV & OR-Tools | May the stars be ever in your favor ⭐</footer>
  </div>
</body>
</html>
'''

@app.route('/', methods=['GET','POST'])
def index():
    output_img = None
    if request.method == 'POST':
        file = request.files['file']
        stars = int(request.form['stars'])
        size = int(request.form['size'])
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        regions = detect_regions_from_image(img, (size,size))
        solution = solve_star_battle(regions, stars)
        # draw stars
        h,w = img.shape[:2]
        cell_h = h/size; cell_w = w/size
        for r,c in solution:
            cy = int((r+0.5)*cell_h); cx = int((c+0.5)*cell_w)
            cv2.circle(img, (cx,cy), int(min(cell_h,cell_w)/3), (0,0,255), thickness=2)
        # save to temp
        fname = next(tempfile._get_candidate_names()) + '.png'
        path = os.path.join(tempfile.gettempdir(), fname)
        cv2.imwrite(path, img)
        output_img = fname
    return render_template_string(TEMPLATE, output_img=output_img)

@app.route('/result/<filename>')
def result(filename):
    path = os.path.join(tempfile.gettempdir(), filename)
    return send_file(path, mimetype='image/png')

if __name__ == '__main__':
    # For local development: override port via CLI
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5000
    print(f"Starting development server on port {port}...")
    # Disable debug mode for production hosting (use Gunicorn or similar WSGI server)
    app.run(host='0.0.0.0', port=port, debug=True)

# Expose the Flask app for WSGI servers (Gunicorn/uwsgi)
application = app



