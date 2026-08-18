"""
    Q-learning trainer and visualizer for the maze in ``map.txt``.

    The file is arranged from the learning logic to the user interface:
    maze rules -> Q-learning -> image output -> Tkinter application.
"""

from PIL import Image, ImageDraw, ImageFont, ImageTk
from urllib.request import Request, urlopen
from tkinter import messagebox, ttk
from urllib.parse import urlencode
from urllib.error import URLError
from pathlib import Path
import tkinter as tk
import threading
import random
import queue
import json
import math
import time
import sys


# Training configuration
TRAINING_EPISODES = 10000
LEARNING_RATE = 0.10
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.9995
MAX_STEPS_PER_EPISODE = 100
RANDOM_SEED = 8

# Training visualization (learner.py never launches maze.py)
SHOW_TRAINING_ON_MAZE = True
MAZE_API_URL = "http://127.0.0.1:8000"
SHOW_EVERY_EPISODES = 100
MOVE_DELAY = 0.08

# Rewards and game rules (matching maze.py where applicable)
STEP_REWARD = -1.0
WALL_REWARD = -5.0
FOOD_REWARD = 30.0
TRAP_REWARD = -30.0
OUT_OF_ENERGY_REWARD = -100.0
START_ENERGY = 20
WALL_ENERGY_COST = 5

SOURCE_ROOT = Path(__file__).resolve().parent

if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
    RESOURCE_ROOT = Path(sys._MEIPASS)
    MAP_FILE = RESOURCE_ROOT / "map.txt"
    IMAGE_DIR = RESOURCE_ROOT / "Images"
else:
    ROOT = SOURCE_ROOT
    MAP_FILE = ROOT / "map.txt"
    IMAGE_DIR = ROOT.parent / "迷路ゲーム" / "Images"

Q_TABLE_IMAGE = ROOT / "q_table.png"
ROUTE_IMAGE = ROOT / "route.png"
HEATMAP_IMAGE = ROOT / "visitation_heatmap.png"
GENERATED_IMAGES = (Q_TABLE_IMAGE, ROUTE_IMAGE, HEATMAP_IMAGE)
TILE_SIZE = 200

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")
TILE_IMAGES = {
    "#": "wall.png",
    ".": "none.png",
    "P": "play.png",
    "F": "food.png",
    "T": "trap.png",
}


def load_map() -> tuple[list[str], tuple[int, int]]:

    """ Read the maze and return it together with its single start position. """

    maze = [line.strip() for line in MAP_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not maze or any(len(row) != len(maze[0]) for row in maze):
        raise ValueError("map.txt must be a non-empty rectangle")

    starts = [(r, c) for r, row in enumerate(maze) for c, tile in enumerate(row) if tile == "P"]

    if len(starts) != 1:
        raise ValueError("map.txt must contain exactly one P")

    return maze, starts[0]


def move(
    maze: list[str],
    state: tuple[int, int],
    action: int, energy: int
) -> tuple[tuple[int, int], int, float, bool]:
    
    """ Apply one action and return ``(position, energy, reward, finished)``. """

    dr, dc = ACTIONS[action]
    row, column = state
    target = (row + dr, column + dc)
    tile = maze[target[0]][target[1]]

    if tile == "#":
        next_state, energy, reward = state, energy - WALL_ENERGY_COST, WALL_REWARD
    else:
        next_state, energy, reward = target, energy - 1, STEP_REWARD

    if energy <= 0:
        return next_state, energy, OUT_OF_ENERGY_REWARD, True
    if tile == "F":
        return next_state, energy, FOOD_REWARD, True
    if tile == "T":
        return next_state, energy, TRAP_REWARD, True
    
    return next_state, energy, reward, False


def choose_action(
        values: list[float],
        epsilon: float,
        rng: random.Random
    ) -> int:

    """ Choose an action using the epsilon-greedy strategy. """

    if rng.random() < epsilon:
        return rng.randrange(len(ACTIONS))

    return best_action(values, rng)


def train(
    maze: list[str],
    start: tuple[int, int],
    progress_callback=None,
    stop_event: threading.Event | None = None,
) -> tuple[dict[tuple[int, int], list[float]], dict[tuple[int, int], int]]:

    """ Learn a Q-table and count how often each maze cell is visited. """

    q_table = {
        (r, c): [0.0] * len(ACTIONS)
        for r, row in enumerate(maze)
        for c, tile in enumerate(row)
        if tile != "#"
    }

    visits = {state: 0 for state in q_table}
    rng = random.Random(RANDOM_SEED)
    epsilon = EPSILON_START
    show_interval = max(1, SHOW_EVERY_EPISODES)

    for episode in range(TRAINING_EPISODES):

        if stop_event and stop_event.is_set():
            raise TrainingStopped(episode, TRAINING_EPISODES)
        state, energy = start, START_ENERGY
        visits[state] += 1
        show_episode = SHOW_TRAINING_ON_MAZE and (
            (episode + 1) % show_interval == 0
            or episode == TRAINING_EPISODES - 1
        )
        if show_episode:
            reset_live_maze()

        for _ in range(MAX_STEPS_PER_EPISODE):
            if stop_event and stop_event.is_set():
                raise TrainingStopped(episode + 1, TRAINING_EPISODES)

            # A displayed episode demonstrates the best policy learned so far.
            action = best_action(q_table[state], rng) if show_episode else choose_action(
                q_table[state], epsilon, rng
            )
            next_state, energy, reward, done = move(maze, state, action, energy)

            if show_episode:
                move_live_maze(ACTION_NAMES[action])
                time.sleep(MOVE_DELAY)

            # The Q-learning update: new knowledge moves the old value toward
            # the immediate reward plus the best value of the next state.
            future = 0.0 if done else max(q_table[next_state])
            old_value = q_table[state][action]
            q_table[state][action] += LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * future - old_value
            )
            state = next_state
            visits[state] += 1
            if done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        report_every = max(1, TRAINING_EPISODES // 100)
        should_report = episode == 0 or (episode + 1) % report_every == 0
        if progress_callback and should_report:
            progress_callback(episode + 1, TRAINING_EPISODES)

    return q_table, visits


class TrainingStopped(Exception):

    """ Raised when the user requests that training stop. """

    def __init__(self, current_episode: int, total_episodes: int):
        super().__init__("training stopped")
        self.current_episode = current_episode
        self.total_episodes = total_episodes


def best_action(
    values: list[float],
    rng: random.Random | None = None
) -> int:

    """ Return one of the highest-valued actions, breaking ties at random. """

    best = max(values)
    choices = [action for action, value in enumerate(values) if math.isclose(value, best)]

    return (rng or random).choice(choices)


def follow_policy(
    maze: list[str],
    start: tuple[int, int],
    q_table: dict[tuple[int, int], list[float]]
) -> tuple[list[tuple[int, int]], list[str], str]:

    """ Follow the learned policy and describe the resulting route. """

    state, energy = start, START_ENERGY
    route, actions, seen = [state], [], {(state, energy)}

    for _ in range(MAX_STEPS_PER_EPISODE):

        action = best_action(q_table[state], random.Random(RANDOM_SEED))
        state, energy, _, done = move(maze, state, action, energy)
        actions.append(ACTION_NAMES[action])
        route.append(state)
        tile = maze[state[0]][state[1]]

        if done:
            return route, actions, "food found" if tile == "F" and energy > 0 else "failed"
        marker = (state, energy)

        if marker in seen:
            return route, actions, "policy loop"
        seen.add(marker)

    return route, actions, "step limit"


def image_font(size: int, bold=False):

    """ Load a clear TrueType font, with a Pillow default fallback. """

    candidates = (
        Path("C:/Windows/Fonts/seguisb.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )

    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue

    return ImageFont.load_default()


def png_map(maze: list[str]) -> Image.Image:

    """ Build a full-size maze image from the individual tile images. """

    image = Image.new("RGBA", (len(maze[0]) * TILE_SIZE, len(maze) * TILE_SIZE), "white")
    tiles = {}

    for tile, filename in TILE_IMAGES.items():
        with Image.open(IMAGE_DIR / filename) as source:
            tiles[tile] = source.convert("RGBA").resize(
                (TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS
            )

    for row, line in enumerate(maze):
        for column, tile in enumerate(line):
            image.alpha_composite(tiles[tile], (column * TILE_SIZE, row * TILE_SIZE))

    return image


def save_q_table(
    maze: list[str],
    q_table: dict[tuple[int, int], list[float]]
) -> None:

    """ Save each state's four Q-values on top of the maze. """

    image = png_map(maze)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    offsets = ((100, 30), (145, 105), (100, 174), (55, 105))
    arrows = ("↑", "→", "↓", "←")
    regular_font = image_font(20)
    bold_font = image_font(21, bold=True)

    for (row, column), values in q_table.items():

        if maze[row][column] in "FT":
            continue
        x, y = column * TILE_SIZE, row * TILE_SIZE
        overlay_draw.rectangle((x, y, x + TILE_SIZE, y + TILE_SIZE), fill=(255, 255, 255, 190))
        winner = best_action(values, random.Random(0))

        for action, value in enumerate(values):

            ox, oy = offsets[action]
            color = "#006d2c" if action == winner else ("#9b1c1c" if value < 0 else "#333333")
            overlay_draw.text(
                (x + ox, y + oy), f"{arrows[action]} {value:.1f}",
                anchor="mm", fill=color,
                font=bold_font if action == winner else regular_font,
            )

        overlay_draw.rectangle((x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1), outline="#888888", width=1)

    Image.alpha_composite(image, overlay).convert("RGB").save(Q_TABLE_IMAGE, "PNG")


def save_route(
    maze: list[str],
    route: list[tuple[int, int]],
    result: str
) -> None:

    """ Save the route selected by the learned policy. """

    image = png_map(maze).convert("RGB")
    draw = ImageDraw.Draw(image)
    center = TILE_SIZE // 2
    points = [(column * TILE_SIZE + center, row * TILE_SIZE + center) for row, column in route]

    if len(points) > 1:
        draw.line(points, fill="white", width=24, joint="curve")
        draw.line(points, fill="#0969da", width=12, joint="curve")

    number_font = image_font(22, bold=True)

    for step, (x, y) in enumerate(points):
        draw.ellipse((x - 26, y - 26, x + 26, y + 26), fill="#0969da", outline="white", width=4)
        draw.text((x, y), str(step), anchor="mm", fill="white", font=number_font)

    draw.text(
        (20, 20), f"Result: {result}", anchor="la", fill="white",
        stroke_width=3, stroke_fill="black", font=image_font(30, bold=True),
    )

    image.save(ROUTE_IMAGE, "PNG")


def save_heatmap(
    maze: list[str],
    visits: dict[tuple[int, int], int]
) -> None:

    """ Save a heatmap showing how frequently each state was visited. """

    image = png_map(maze)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    largest = max(visits.values(), default=1)
    scale = math.log1p(largest)
    count_font = image_font(24, bold=True)

    for (row, column), count in visits.items():
        intensity = math.log1p(count) / scale if scale else 0.0
        red = round(30 + 225 * intensity)
        green = round(110 - 80 * intensity)
        blue = round(220 - 190 * intensity)
        x, y = column * TILE_SIZE, row * TILE_SIZE
        draw.rectangle((x, y, x + TILE_SIZE, y + TILE_SIZE), fill=(red, green, blue, 170))
        draw.text(
            (x + TILE_SIZE // 2, y + TILE_SIZE // 2), f"{count:,}",
            anchor="mm", fill="white", stroke_width=2, stroke_fill="black", font=count_font,
        )

    draw.text(
        (20, 20), "Blue: rarely visited  ·  Red: commonly visited",
        anchor="la", fill="white", stroke_width=3, stroke_fill="black",
        font=image_font(28, bold=True),
    )

    Image.alpha_composite(image, overlay).convert("RGB").save(HEATMAP_IMAGE, "PNG")


def post_to_maze(endpoint: str) -> dict:

    """ Send a command to the running maze application. """

    try:
        request = Request(f"{MAZE_API_URL.rstrip('/')}/{endpoint}", data=b"", method="POST")
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))

    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            f"maze.py（{MAZE_API_URL}）に接続できません。"
            "maze.pyのHTTPサーバーを開始してください。"
        ) from error


def update_maze_rewards() -> None:

    """ Make the maze application's rewards match this trainer's settings. """

    parameters = urlencode(
        {
            "step": STEP_REWARD,
            "wall": WALL_REWARD,
            "food": FOOD_REWARD,
            "trap": TRAP_REWARD,
            "out_of_energy": OUT_OF_ENERGY_REWARD,
        }
    )
    response = post_to_maze(f"rewards?{parameters}")

    if not response.get("ok"):
        raise RuntimeError(response.get("error", "maze.pyの報酬設定を更新できませんでした。"))


def reset_live_maze() -> None:

    """ Reset the maze before displaying an evaluation episode. """

    response = post_to_maze("reset")

    if not response.get("ok"):
        raise RuntimeError(response.get("error", "迷路のリセットが拒否されました。"))


def move_live_maze(action: str) -> None:

    """ Display one training action in the maze application. """

    response = post_to_maze(f"move?direction={action}")

    if response.get("error"):
        raise RuntimeError(
            f"maze.pyがリモート訓練を拒否しました: {response['error']}。"
            "maze.pyのHTTP設定タブでリモートプレイを許可してください。"
        )


def run_training(progress_callback=None, stop_event=None) -> dict:

    """ Run learning and return everything needed for the result views. """

    maze, start = load_map()
    q_table, visits = train(maze, start, progress_callback, stop_event)
    route, actions, result = follow_policy(maze, start, q_table)

    return {
        "maze": maze, "q_table": q_table, "visits": visits,
        "route": route, "actions": actions, "result": result,
    }


def delete_generated_images() -> list[str]:

    """ Delete learner output images and return any cleanup errors. """

    errors = []
    for filename in GENERATED_IMAGES:
        try:
            filename.unlink(missing_ok=True)
        except OSError as error:
            errors.append(f"{filename.name}: {error}")

    return errors


class PngView(ttk.Frame):

    """ Display a PNG fitted to the available tab area. """

    def __init__(self, parent, filename: Path):

        super().__init__(parent)
        self.filename = filename
        self.source_image: Image.Image | None = None
        self.display_image: ImageTk.PhotoImage | None = None
        self.resize_job: str | None = None
        self.zoom_factor = 1.0
        self.zoom_label = tk.StringVar(value="全体表示 100%")

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(toolbar, text="縮小 −", command=lambda: self._zoom(1 / 1.25)).pack(side="left")
        ttk.Button(toolbar, text="拡大 ＋", command=lambda: self._zoom(1.25)).pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="全体表示", command=self._reset_zoom).pack(side="left", padx=(6, 0))
        ttk.Label(toolbar, textvariable=self.zoom_label).pack(side="left", padx=(12, 0))

        self.canvas = tk.Canvas(self, background="#eceff3", highlightthickness=0)
        x_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        y_scroll.grid(row=1, column=1, sticky="ns")
        x_scroll.grid(row=2, column=0, sticky="ew")
        self.canvas.bind("<Configure>", self._schedule_fit)
        self.canvas.bind("<Control-MouseWheel>", self._mouse_zoom)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.show_placeholder()

    def show_placeholder(self, text="「開始」を押すと画像が表示されます。"):

        """ Replace the current image with a short status message. """

        self.canvas.delete("all")
        self.source_image = None
        self.display_image = None
        self.canvas.create_text(28, 28, text=text, anchor="nw", fill="#52606d", font=("Yu Gothic UI", 11))

    def reload(self):

        """ Reload the PNG from disk and fit it inside the canvas. """

        if not self.filename.exists():
            self.show_placeholder(f"{self.filename.name} はまだ作成されていません。")
            return

        try:
            with Image.open(self.filename) as source:
                self.source_image = source.convert("RGB")
            self.zoom_factor = 1.0
            self._fit_image()
        except Exception as error:
            self.show_placeholder(f"{self.filename.name} を表示できません。\n{error}")

    def _schedule_fit(self, _event=None):

        """ Debounce resize events so large images are not resized repeatedly. """

        if self.resize_job:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(80, self._fit_image)

    def _zoom(self, multiplier):

        """ Change the user-controlled zoom while keeping it in a useful range. """

        if self.source_image is None:
            return
        self.zoom_factor = min(4.0, max(0.5, self.zoom_factor * multiplier))
        self._fit_image()

    def _reset_zoom(self):

        """ Return to the fit-to-window zoom level. """

        self.zoom_factor = 1.0
        self._fit_image()

    def _mouse_zoom(self, event):

        """ Zoom when Ctrl + mouse wheel is used over the canvas. """

        self._zoom(1.25 if event.delta > 0 else 1 / 1.25)

        return "break"

    def _fit_image(self):

        """ Resize and center the source image for the current canvas size. """

        self.resize_job = None
        if self.source_image is None:
            return

        available_width = max(1, self.canvas.winfo_width() - 20)
        available_height = max(1, self.canvas.winfo_height() - 20)
        fit_scale = min(
            available_width / self.source_image.width,
            available_height / self.source_image.height,
        )
        scale = fit_scale * self.zoom_factor
        width = max(1, round(self.source_image.width * scale))
        height = max(1, round(self.source_image.height * scale))
        resized = self.source_image.resize(
            (width, height), Image.Resampling.LANCZOS,
        )
        self.display_image = ImageTk.PhotoImage(resized, master=self.canvas)
        self.canvas.delete("all")
        region_width = max(self.canvas.winfo_width(), width)
        region_height = max(self.canvas.winfo_height(), height)
        self.canvas.configure(scrollregion=(0, 0, region_width, region_height))
        self.canvas.create_image(
            region_width // 2, region_height // 2,
            image=self.display_image, anchor="center",
        )
        self.zoom_label.set(f"全体表示の {round(self.zoom_factor * 100)}%")

        if width > self.canvas.winfo_width():
            self.canvas.xview_moveto((region_width - self.canvas.winfo_width()) / (2 * region_width))
        if height > self.canvas.winfo_height():
            self.canvas.yview_moveto((region_height - self.canvas.winfo_height()) / (2 * region_height))

class LearnerApp(tk.Tk):

    """ Tkinter window used to configure, run, and inspect training. """

    PARAMETER_FIELDS = (
        ("訓練エピソード数", "TRAINING_EPISODES", int),
        ("移動報酬", "STEP_REWARD", float),
        ("壁に衝突したときの報酬", "WALL_REWARD", float),
        ("食べ物を取ったときの報酬", "FOOD_REWARD", float),
        ("罠に入ったときの報酬", "TRAP_REWARD", float),
        ("エネルギー切れの報酬", "OUT_OF_ENERGY_REWARD", float),
    )

    def __init__(self):

        super().__init__()
        self.title("Qラーニング")
        # maze.pyと同じ最小ウィンドウサイズにする。
        self.minsize(260, 220)
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.parameter_vars: dict[str, tk.StringVar] = {}
        self.url_var = tk.StringVar(value=MAZE_API_URL)
        self.live_var = tk.BooleanVar(value=SHOW_TRAINING_ON_MAZE)
        self.status_var = tk.StringVar(value="準備完了")
        self.progress_var = tk.DoubleVar(value=0)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        settings_tab = ttk.Frame(notebook, padding=18)
        notebook.add(settings_tab, text="訓練・接続設定")
        self.views = (
            PngView(notebook, Q_TABLE_IMAGE),
            PngView(notebook, ROUTE_IMAGE),
            PngView(notebook, HEATMAP_IMAGE),
        )
        notebook.add(self.views[0], text="Qテーブル")
        notebook.add(self.views[1], text="経路")
        notebook.add(self.views[2], text="訪問ヒートマップ")
        self._build_settings(settings_tab)
        for view in self.views:
            view.reload()
        self.after(100, self._process_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_settings(self, parent):

        """ Create the settings tab. """
        parameters = ttk.LabelFrame(parent, text="パラメータ・報酬", padding=14)
        parameters.pack(fill="x")
        for row, (label, name, _converter) in enumerate(self.PARAMETER_FIELDS):
            ttk.Label(parameters, text=label).grid(row=row, column=0, sticky="w", pady=5)
            variable = tk.StringVar(value=str(globals()[name]))
            self.parameter_vars[name] = variable
            ttk.Entry(parameters, textvariable=variable, width=18).grid(row=row, column=1, sticky="w", padx=(18, 0), pady=5)
        parameters.columnconfigure(2, weight=1)

        connection = ttk.LabelFrame(parent, text="maze.py HTTP接続", padding=14)
        connection.pack(fill="x", pady=(14, 0))
        ttk.Label(connection, text="接続先URL").grid(row=0, column=0, sticky="w")
        ttk.Entry(connection, textvariable=self.url_var).grid(row=0, column=1, sticky="ew", padx=(18, 8))
        ttk.Checkbutton(
            connection,
            text="評価エピソードを迷路ゲームに表示する",
            variable=self.live_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        connection.columnconfigure(1, weight=1)

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(18, 0))
        self.start_button = ttk.Button(actions, text="開始", command=self._start_training)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="停止", command=self._stop_training, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Progressbar(actions, variable=self.progress_var, maximum=100).pack(side="left", fill="x", expand=True, padx=14)
        ttk.Label(actions, textvariable=self.status_var).pack(side="right")

        note = ("開始すると、入力した報酬設定を起動中のmaze.pyへ送信します。"
                "評価エピソードを表示する場合は、maze.pyのHTTPサーバーを開始し、"
                "リモートプレイを許可してください。")
        ttk.Label(parent, text=note, wraplength=820, foreground="#52606d", justify="left").pack(fill="x", pady=(14, 0))

    def _apply_settings(self):

        """ Validate the form and copy its values into the training settings. """

        global MAZE_API_URL, SHOW_TRAINING_ON_MAZE
        values = {}

        for label, name, converter in self.PARAMETER_FIELDS:
            try:
                value = converter(self.parameter_vars[name].get())
            except ValueError as error:
                raise ValueError(f"「{label}」に正しい数値を入力してください。") from error
            if name == "TRAINING_EPISODES" and value <= 0:
                raise ValueError("訓練エピソード数は1以上にしてください。")
            if not math.isfinite(value):
                raise ValueError(f"「{label}」には有限の数値を入力してください。")
            values[name] = value
        url = self.url_var.get().strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("接続先URLは http:// または https:// で始めてください。")
        globals().update(values)
        MAZE_API_URL = url
        SHOW_TRAINING_ON_MAZE = self.live_var.get()

    def _start_training(self):

        """ Validate settings and start training on a background thread. """

        try:
            self._apply_settings()
        except ValueError as error:
            messagebox.showerror("設定エラー", str(error), parent=self)
            return

        self.stop_event = threading.Event()
        self.progress_var.set(0)
        self._set_training_state(running=True, status="訓練中…")

        def progress(done, total):

            """ Send progress safely from the worker thread to Tkinter. """

            self.events.put(("progress", done, total))

        def worker():

            """ Run slow network and training work outside Tkinter's thread. """

            try:
                try:
                    update_maze_rewards()
                except RuntimeError:
                    if SHOW_TRAINING_ON_MAZE:
                        raise
                payload = run_training(progress, self.stop_event)
                self.events.put(("complete", payload, payload["result"]))
            except TrainingStopped as stopped:
                self.events.put(
                    ("stopped", stopped.current_episode, stopped.total_episodes)
                )
            except Exception as error:
                self.events.put(("error", str(error), None))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_training(self):

        """ Ask the background training loop to stop safely. """

        self.stop_event.set()
        self.stop_button.configure(state="disabled")
        self.status_var.set("停止処理中…")

    def _set_training_state(self, running: bool, status: str) -> None:

        """ Update the controls shared by every completion/error path. """

        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.status_var.set(status)

    def _on_close(self):

        """ Stop training, release images, and remove generated files. """

        self.stop_event.set()

        for view in self.views:
            view.source_image = None
            view.display_image = None
        errors = delete_generated_images()

        if errors:
            messagebox.showwarning(
                "画像削除エラー",
                "次の画像を削除できませんでした。\n" + "\n".join(errors),
                parent=self,
            )
        self.destroy()

    def _process_events(self):

        """ Transfer progress and results from the worker thread to Tkinter. """

        try:
            while True:
                kind, first, second = self.events.get_nowait()
                if kind == "progress":
                    self.progress_var.set(first * 100 / second)
                    self.status_var.set(f"訓練中… {first:,}/{second:,}")
                elif kind == "complete":
                    result_label = {
                        "food found": "食べ物を取得",
                        "failed": "失敗",
                        "policy loop": "方策がループ",
                        "step limit": "ステップ上限",
                    }.get(second, second)
                    action_labels = {"up": "上", "right": "右", "down": "下", "left": "左"}
                    self.status_var.set("PNG画像を作成中…")
                    self.update_idletasks()
                    try:
                        save_q_table(first["maze"], first["q_table"])
                        save_route(first["maze"], first["route"], first["result"])
                        save_heatmap(first["maze"], first["visits"])
                    except Exception as error:
                        self._set_training_state(False, "画像の作成に失敗しました")
                        messagebox.showerror("画像作成エラー", str(error), parent=self)
                        continue
                    for view in self.views:
                        view.reload()
                    self.progress_var.set(100)
                    self._set_training_state(False, f"完了: {result_label}")
                    messagebox.showinfo(
                        "訓練完了",
                        f"結果: {result_label}\n"
                        f"経路: {' → '.join(action_labels.get(action, action) for action in first['actions']) or '（なし）'}\n\n"
                        "3つのPNG画像を更新しました。",
                        parent=self,
                    )
                elif kind == "stopped":
                    self.progress_var.set(first * 100 / second)
                    self._set_training_state(False, f"停止: {first:,} / {second:,}")
                elif kind == "error":
                    self._set_training_state(False, "訓練に失敗しました")
                    messagebox.showerror("訓練エラー", first, parent=self)
        except queue.Empty:
            pass
        self.after(100, self._process_events)


def main() -> None:

    """ Open the Q-learning application. """

    LearnerApp().mainloop()


if __name__ == "__main__":

    main()
