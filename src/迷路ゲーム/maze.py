"""
    Tkinter maze game with a Japanese interface and an HTTP API.

    map.txtを編集して「マップ再読込」またはF5を押すと迷路を変更できます。
    マップは長方形で、外周が壁で囲まれ、プレイヤー(P)が1人だけ必要です。

    The file is arranged from the maze rules to the user interface:
    configuration -> HTTP API -> sound and images -> Tkinter application.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from tkinter import font as tkfont
from tkinter import messagebox
from pathlib import Path
from tkinter import ttk
import tkinter as tk
import threading
import winsound
import struct
import json
import math
import io
import wave


ROOT = Path(__file__).resolve().parent
MAP_FILE = ROOT / "map.txt"
IMAGE_DIR = ROOT / "Images"

# Game rules
TILE_SIZE = 50
FOOD_SCORE = 30
START_ENERGY = 20
WALL_PENALTY = 5
DEFAULT_REWARDS = {
    "step": -1.0,
    "wall": -5.0,
    "food": 30.0,
    "trap": -30.0,
    "out_of_energy": -100.0,
}

# Sound played when a game ends
WIN_MELODY = (
    (523, 100),  # C5
    (659, 100),  # E5
    (784, 120),  # G5
    (1047, 300),  # C6
)
SOUND_SAMPLE_RATE = 44100
SOUND_VOLUME = 0.25
NOTE_GAP_MS = 25

TILE_FILES = {
    "#": "wall.png",
    ".": "none.png",
    "P": "play.png",
    "F": "food.png",
    "T": "trap.png",
}
VALID_TILES = set(TILE_FILES)
JAPANESE_FONT_CANDIDATES = (
    "Yu Gothic UI",
    "Yu Gothic",
    "Meiryo UI",
    "Meiryo",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Hiragino Sans",
)


class MazeRequestHandler(BaseHTTPRequestHandler):

    """ Serve the maze state and controls as JSON over HTTP. """

    def do_GET(self):

        """ Return API information or the current game state. """

        path = urlparse(self.path).path
        if path == "/":
            self.send_json(self.server.app.get_api_info())
            return
        if path == "/state":
            self.send_json(self.server.app.get_remote_state())
            return
        self.send_error(404)

    def do_POST(self):

        """ Apply a remote move, reset, or reward update. """

        parsed = urlparse(self.path)
        values = parse_qs(parsed.query)
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            body = self.rfile.read(content_length).decode("utf-8")
            for key, value in parse_qs(body).items():
                values.setdefault(key, value)

        if parsed.path == "/move":
            direction = self.first_value(values, "direction")
            result = self.server.app.remote_move(direction)
            self.send_json(result)
            return
        if parsed.path == "/move_to":
            result = self.server.app.remote_move_to_cell(
                self.first_value(values, "row"),
                self.first_value(values, "column"),
            )
            self.send_json(result)
            return
        if parsed.path == "/reset":
            self.send_json(self.server.app.remote_reset())
            return
        if parsed.path == "/rewards":
            reward_values = {
                name: self.first_value(values, name)
                for name in DEFAULT_REWARDS
            }
            self.send_json(self.server.app.remote_set_rewards(reward_values))
            return
        self.send_error(404)

    @staticmethod
    def first_value(values: dict[str, list[str]], key: str) -> str:

        """ Return the first parsed query value, or an empty string. """

        value = values.get(key, [""])
        return value[0] if value else ""

    def send_json(self, payload: dict):

        """ Send one successful JSON response. """

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):

        """ Suppress the request handler's default console log. """

        return


class MazeApp(tk.Tk):

    """ Display and control the maze game. """

    def __init__(self):

        super().__init__()
        self.ui_thread = threading.current_thread()
        self.configure_japanese_fonts()
        self.title("迷路")
        self.source_images = self.load_images()
        self.source_pixels = self.read_source_pixels()
        self.source_transparency = self.read_source_transparency()
        self.scaled_images: dict[tuple[int, int], dict[str, tk.PhotoImage]] = {}
        self.maze: list[list[str]] = []
        self.player = (0, 0)
        self.game_over = False
        self.terminal_result: str | None = None
        self.terminal_reason: str = ""
        self.result_window: tk.Toplevel | None = None
        self.resize_job: str | None = None
        self.no_steps: int = 0
        self.remaining_energy: int = START_ENERGY
        self.reward_values = DEFAULT_REWARDS.copy()
        self.last_reward: float = 0.0
        self.total_reward: float = 0.0
        self.http_server: ThreadingHTTPServer | None = None
        self.http_thread: threading.Thread | None = None
        self.board_geometry = (0, 0, TILE_SIZE, TILE_SIZE, 0, 0)
        self.self_play_enabled = tk.BooleanVar(value=True)
        self.remote_play_enabled = tk.BooleanVar(value=False)
        self.http_host = tk.StringVar(value="127.0.0.1")
        self.http_port = tk.StringVar(value="8000")
        self.http_status = tk.StringVar(value="HTTPサーバー停止中")
        self.remote_url = tk.StringVar(value="http://localhost:8000/")
        self.reward_status = tk.StringVar(value="累計報酬: 0.0")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        game_tab = tk.Frame(notebook)
        network_tab = tk.Frame(notebook, padx=12, pady=12)
        notebook.add(game_tab, text="ゲーム")
        notebook.add(network_tab, text="HTTP設定")

        controls = tk.Frame(game_tab, padx=10, pady=8)
        controls.pack(fill="x")
        tk.Button(controls, text="マップ再読込(F5)", command=self.reload_map).pack(side="left")
        self.status = tk.StringVar(value="マップファイルから迷路を読み込んでください")
        tk.Label(controls, textvariable=self.status, anchor="w").pack(side="left", padx=10)
        tk.Label(controls, textvariable=self.reward_status, anchor="e").pack(side="right")

        main_area = tk.Frame(game_tab)
        main_area.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(main_area, highlightthickness=0, background="white")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self.schedule_draw_maze)
        self.canvas.bind("<Button-1>", self.handle_canvas_click)

        log_panel = tk.Frame(main_area, padx=8)
        log_panel.pack(side="right", fill="y")
        tk.Label(log_panel, text="ログ", anchor="w").pack(fill="x")

        log_body = tk.Frame(log_panel)
        log_body.pack(fill="both", expand=True)
        self.log_display = tk.Text(
            log_body,
            width=34,
            height=12,
            wrap="word",
            state="disabled",
            relief="sunken",
            borderwidth=1,
        )
        log_scrollbar = tk.Scrollbar(log_body, orient="vertical", command=self.log_display.yview)
        self.log_display.configure(yscrollcommand=log_scrollbar.set)
        self.log_display.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")

        self.build_network_tab(network_tab)

        self.bind("<F5>", lambda _event: self.reload_map())
        self.bind("<Up>", lambda _event: self.move_player(-1, 0, source="self"))
        self.bind("<Down>", lambda _event: self.move_player(1, 0, source="self"))
        self.bind("<Left>", lambda _event: self.move_player(0, -1, source="self"))
        self.bind("<Right>", lambda _event: self.move_player(0, 1, source="self"))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.reload_map()

    def build_network_tab(self, parent: tk.Frame):

        """ Build the remote-play settings tab. """

        mode_box = ttk.LabelFrame(parent, text="プレイ権限", padding=10)
        mode_box.pack(fill="x")
        ttk.Checkbutton(
            mode_box,
            text="セルフプレイを許可（キーボード・キャンバスクリック）",
            variable=self.self_play_enabled,
        ).pack(anchor="w")
        ttk.Checkbutton(
            mode_box,
            text="リモートプレイを許可（HTTPから移動操作）",
            variable=self.remote_play_enabled,
        ).pack(anchor="w", pady=(6, 0))

        server_box = ttk.LabelFrame(parent, text="HTTPアクセス", padding=10)
        server_box.pack(fill="x", pady=(12, 0))
        fields = tk.Frame(server_box)
        fields.pack(fill="x")
        ttk.Label(fields, text="ホスト").grid(row=0, column=0, sticky="w")
        ttk.Entry(fields, textvariable=self.http_host, width=18).grid(row=0, column=1, padx=(8, 18), sticky="w")
        ttk.Label(fields, text="ポート").grid(row=0, column=2, sticky="w")
        ttk.Entry(fields, textvariable=self.http_port, width=8).grid(row=0, column=3, padx=(8, 0), sticky="w")
        fields.columnconfigure(4, weight=1)

        buttons = tk.Frame(server_box)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="開始", command=self.start_http_server).pack(side="left")
        ttk.Button(buttons, text="停止", command=self.stop_http_server).pack(side="left", padx=(8, 0))

        ttk.Label(server_box, textvariable=self.http_status, anchor="w").pack(fill="x", pady=(10, 0))
        ttk.Label(server_box, textvariable=self.remote_url, anchor="w").pack(fill="x", pady=(4, 0))

        endpoints = ttk.LabelFrame(parent, text="相手に共有する機能", padding=10)
        endpoints.pack(fill="both", expand=True, pady=(12, 0))
        endpoint_text = tk.Text(endpoints, height=8, wrap="word", relief="flat", background=parent.cget("background"))
        endpoint_text.insert(
            "1.0",
            "AIプレイヤーはHTTP APIで盤面取得と移動操作を行います。\n"
            "ゲーム終了時はレスポンスに勝敗、残りエネルギー、リセットコマンドが入ります。\n\n"
            "API:\n"
            "GET /state\n"
            "POST /move?direction=up|down|left|right\n"
            "POST /move_to?row=0&column=0\n"
            "POST /reset\n"
            "POST /rewards?step=-1&wall=-5&food=30&trap=-30&out_of_energy=-100",
        )
        endpoint_text.configure(state="disabled")
        endpoint_text.pack(fill="both", expand=True)

    def call_on_ui(self, callback):

        """ Run a callback on Tkinter's thread and return its result. """

        if threading.current_thread() is self.ui_thread:
            return callback()

        done = threading.Event()
        result = {"value": None, "error": None}

        def run_callback():

            """ Capture the callback result for the waiting HTTP thread. """

            try:
                result["value"] = callback()
            except Exception as error:
                result["error"] = error
            finally:
                done.set()

        self.after(0, run_callback)
        if not done.wait(5):
            raise TimeoutError("UI thread did not respond")
        if result["error"] is not None:
            raise result["error"]
        return result["value"]

    def start_http_server(self):

        """ Validate the address and start the remote-play server. """

        if self.http_server is not None:
            self.http_status.set("HTTPサーバーは起動中です")
            return

        host = self.http_host.get().strip()
        if not host:
            messagebox.showerror("HTTP設定エラー", "ホストを入力してください。")
            return
        try:
            port = int(self.http_port.get())
            if not (0 <= port <= 65535):
                raise ValueError
        except ValueError:
            messagebox.showerror("HTTP設定エラー", "ポートは0から65535の整数で入力してください。")
            return

        try:
            server = ThreadingHTTPServer((host, port), MazeRequestHandler)
        except OSError as error:
            messagebox.showerror("HTTPサーバーを起動できません", str(error))
            return

        server.app = self
        self.http_server = server
        actual_host, actual_port = server.server_address[:2]
        self.http_port.set(str(actual_port))
        if actual_host in ("0.0.0.0", "::"):
            self.remote_url.set(f"ローカル: http://localhost:{actual_port}/　共有: http://<このPCのIP>:{actual_port}/")
        else:
            self.remote_url.set(f"http://{actual_host}:{actual_port}/")
        self.http_status.set(f"HTTPサーバー起動中: {actual_host}:{actual_port}")
        self.http_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.http_thread.start()

    def stop_http_server(self):

        """ Stop and release the remote-play server. """

        server = self.http_server
        if server is None:
            self.http_status.set("HTTPサーバー停止中")
            return
        self.http_server = None
        self.http_status.set("HTTPサーバー停止中")
        server.shutdown()
        server.server_close()
        self.http_thread = None

    def on_close(self):

        """ Stop background work before closing the application. """

        self.stop_http_server()
        self.destroy()

    def get_api_info(self) -> dict:

        """ Describe the endpoints exposed by the HTTP server. """

        return {
            "message": "Remote API.",
            "endpoints": {
                "state": "GET /state",
                "move": "POST /move?direction=up|down|left|right",
                "move_to_adjacent_cell": "POST /move_to?row=0&column=0",
                "reset": "POST /reset",
                "rewards": "POST /rewards?step=-1&wall=-5&food=30&trap=-30&out_of_energy=-100",
            },
        }

    def get_remote_state(self) -> dict:

        """ Return a thread-safe snapshot of the current game. """

        return self.call_on_ui(self.build_state_snapshot)

    def build_state_snapshot(self) -> dict:

        """ Build the state shared with a remote player. """

        result = self.terminal_result if self.game_over else "playing"
        state = {
            "result": result,
            "game_over": self.game_over,
            "won": result == "win",
            "lost": result == "lose",
            "maze": ["".join(row) for row in self.maze],
            "player": {"row": self.player[0], "column": self.player[1]},
            "steps": self.no_steps,
            "energy_left": self.remaining_energy,
            "remaining_energy": self.remaining_energy,
            "last_reward": self.last_reward,
            "total_reward": self.total_reward,
            "reward_values": self.reward_values.copy(),
            "status": self.status.get(),
            "self_play_enabled": self.self_play_enabled.get(),
            "remote_play_enabled": self.remote_play_enabled.get(),
        }
        if self.game_over:
            state["message"] = self.terminal_reason
            state["reset_command"] = "POST /reset"
        return state

    def build_remote_response(
        self,
        ok: bool,
        error: str | None = None
    ) -> dict:

        """ Add command status to a current game snapshot. """

        state = self.build_state_snapshot()
        response = {
            "ok": ok,
            "result": state["result"],
            "game_over": state["game_over"],
            "energy_left": state["energy_left"],
            "state": state,
        }
        if error is not None:
            response["error"] = error
        if state["game_over"]:
            response["message"] = state["message"]
            response["reset_command"] = state["reset_command"]
        return response

    def remote_reset(self) -> dict:

        """ Reset the maze on Tkinter's thread. """

        def apply_reset():

            """ Reload the map and return its fresh state. """

            self.reload_map()
            return self.build_remote_response(True)

        return self.call_on_ui(apply_reset)

    def remote_set_rewards(self, values: dict[str, str]) -> dict:

        """ Validate and install reward values sent by the learner. """

        try:
            parsed = {name: float(values[name]) for name in DEFAULT_REWARDS}
        except (KeyError, TypeError, ValueError):
            return self.call_on_ui(
                lambda: self.build_remote_response(False, "all reward values must be valid numbers")
            )
        if not all(math.isfinite(value) for value in parsed.values()):
            return self.call_on_ui(
                lambda: self.build_remote_response(False, "reward values must be finite")
            )

        def apply_rewards():

            """ Apply rewards and return them with the current state. """

            self.reward_values.update(parsed)
            response = self.build_remote_response(True)
            response["reward_values"] = self.reward_values.copy()
            return response

        return self.call_on_ui(apply_rewards)

    def record_reward(self, event: str):

        """ Add one game event's reward to the displayed total. """

        self.last_reward = self.reward_values[event]
        self.total_reward += self.last_reward
        self.reward_status.set(f"累計報酬: {self.total_reward:g}")

    def reset_terminal_state(self):

        """ Return the result fields to their active-game values. """

        self.game_over = False
        self.terminal_result = None
        self.terminal_reason = ""

    def finish_game(
        self,
        result: str,
        reason: str,
        source: str,
        popup_detail: str | None = None
    ):

        """ Record a final result and notify a local player. """

        self.game_over = True
        self.terminal_result = result
        self.terminal_reason = reason
        if not self.remote_play_enabled.get():
            self.play_result_sound(result)
        if source != "remote" and not self.remote_play_enabled.get():
            self.show_result_window(result, popup_detail)

    @staticmethod
    def build_melody_wave(melody) -> bytes:

        """ Build one WAV so audio drivers cannot collapse separate beeps. """

        samples = []
        gap_samples = round(SOUND_SAMPLE_RATE * NOTE_GAP_MS / 1000)
        for frequency, duration in melody:
            note_samples = round(SOUND_SAMPLE_RATE * duration / 1000)
            fade_samples = min(round(SOUND_SAMPLE_RATE * 0.008), note_samples // 2)
            for index in range(note_samples):
                envelope = 1.0
                if fade_samples:
                    envelope = min(
                        1.0,
                        index / fade_samples,
                        (note_samples - 1 - index) / fade_samples,
                    )
                value = int(
                    32767
                    * SOUND_VOLUME
                    * envelope
                    * math.sin(2 * math.pi * frequency * index / SOUND_SAMPLE_RATE)
                )
                samples.append(value)
            samples.extend([0] * gap_samples)

        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SOUND_SAMPLE_RATE)
            wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return output.getvalue()

    @classmethod
    def play_result_sound(cls, result: str):

        """ Play the win melody forward or the loss melody backward. """

        def play_melody():

            """ Build and play the result sound outside Tkinter's thread. """

            melody = WIN_MELODY if result == "win" else tuple(reversed(WIN_MELODY))
            sound = cls.build_melody_wave(melody)
            winsound.PlaySound(sound, winsound.SND_MEMORY | winsound.SND_NODEFAULT)

        threading.Thread(target=play_melody, daemon=True).start()

    def should_accept_remote_move(self) -> tuple[bool, str | None]:

        """ Explain whether the API may move the player now. """

        if not self.remote_play_enabled.get():
            return False, "remote play is disabled"
        if self.game_over:
            return False, "game is over; use POST /reset"
        return True, None

    def remote_move(self, direction: str) -> dict:

        """ Translate a named direction into one remote move. """

        moves = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        if direction not in moves:
            return self.call_on_ui(lambda: self.build_remote_response(False, "unknown direction"))

        def apply_move():

            """ Validate and apply the requested move on the UI thread. """

            accepted, error = self.should_accept_remote_move()
            if not accepted:
                return self.build_remote_response(False, error)
            moved = self.move_player(*moves[direction], source="remote")
            return self.build_remote_response(moved)

        return self.call_on_ui(apply_move)

    def remote_move_to_cell(
        self,
        row_value: str,
        column_value: str
    ) -> dict:

        """ Move remotely to an adjacent cell identified by coordinates. """

        try:
            row = int(row_value)
            column = int(column_value)
        except ValueError:
            return self.call_on_ui(lambda: self.build_remote_response(False, "row and column must be integers"))

        def apply_move():

            """ Validate and apply the requested cell on the UI thread. """

            accepted, error = self.should_accept_remote_move()
            if not accepted:
                return self.build_remote_response(False, error)
            moved = self.move_player_to_cell(row, column, source="remote")
            return self.build_remote_response(moved)

        return self.call_on_ui(apply_move)

    def configure_japanese_fonts(self):

        """ Select the first installed font that supports Japanese text. """

        available_fonts = set(tkfont.families(self))
        default_font = tkfont.nametofont("TkDefaultFont")
        default_family = default_font.actual("family")
        family = next(
            (candidate for candidate in JAPANESE_FONT_CANDIDATES if candidate in available_fonts),
            default_family,
        )
        default_font.configure(family=family)
        self.ui_font_family = family
        self.option_add("*Font", default_font)

    @staticmethod
    def load_images() -> dict[str, tk.PhotoImage]:

        """ Load the source image for every map tile. """

        images = {}
        for symbol, filename in TILE_FILES.items():
            image_path = IMAGE_DIR / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"必要な画像ファイルが見つかりません:{image_path}")
            images[symbol] = tk.PhotoImage(file=image_path)
        return images

    def read_source_pixels(self) -> dict[str, list[list[str]]]:

        """ Cache source colors for nearest-neighbor resizing. """

        pixels = {}
        for symbol, image in self.source_images.items():
            rows = []
            for y in range(image.height()):
                row = []
                for x in range(image.width()):
                    red, green, blue = image.get(x, y)
                    row.append(f"#{red:02x}{green:02x}{blue:02x}")
                rows.append(row)
            pixels[symbol] = rows
        return pixels

    def read_source_transparency(self) -> dict[str, list[list[bool]]]:

        """ Cache source transparency for nearest-neighbor resizing. """

        transparency = {}
        for symbol, image in self.source_images.items():
            rows = []
            for y in range(image.height()):
                row = []
                for x in range(image.width()):
                    row.append(image.transparency_get(x, y))
                rows.append(row)
            transparency[symbol] = rows
        return transparency

    def read_map(self) -> list[list[str]]:

        """ Read and validate the maze map. """

        if not MAP_FILE.is_file():
            raise ValueError("map.txtが見つかりません。")
        rows = [line.strip() for line in MAP_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) < 3 or len(rows[0]) < 3:
            raise ValueError("マップは3行3列以上にしてください。")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("すべての行の幅を同じにしてください。")
        bad = sorted(set("".join(rows)) - VALID_TILES)
        if bad:
            raise ValueError(f"使えないマップ記号があります:{','.join(bad)}")
        if sum(row.count("P") for row in rows) != 1:
            raise ValueError("マップにはP(プレイヤー)を1つだけ入れてください。")
        if any(tile != "#" for tile in rows[0] + rows[-1]):
            raise ValueError("上端と下端はすべて壁(#)にしてください。")
        if any(row[0] != "#" or row[-1] != "#" for row in rows):
            raise ValueError("左端と右端はすべて壁(#)にしてください。")
        return [list(row) for row in rows]

    def reload_map(self):

        """ Reload the map and restore the initial game state. """

        self.close_result_window()
        try:
            self.maze = self.read_map()
        except (OSError, ValueError) as error:
            messagebox.showerror("マップを読み込めません", str(error))
            return
        self.player = next(
            (row, column)
            for row, line in enumerate(self.maze)
            for column, tile in enumerate(line)
            if tile == "P"
        )
        self.reset_terminal_state()
        self.no_steps = 0
        self.remaining_energy = START_ENERGY
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.reward_status.set("累計報酬: 0.0")
        self.canvas.config(width=len(self.maze[0]) * TILE_SIZE, height=len(self.maze) * TILE_SIZE)
        self.minsize(260, 220)
        self.draw_maze()
        self.status.set(f"{len(self.maze[0])}x{len(self.maze)}の閉じた迷路　|　矢印キーでプレイヤーを移動　|　最初のエネルギー{START_ENERGY}点")

    def schedule_draw_maze(self, _event=None):

        """ Debounce resize events before redrawing the maze. """

        if not self.maze:
            return
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(30, self.draw_maze)

    def draw_maze(self):

        """ Fit the tile grid to the canvas and draw it. """

        self.resize_job = None
        height, width = len(self.maze), len(self.maze[0])
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = width * TILE_SIZE
        if canvas_height <= 1:
            canvas_height = height * TILE_SIZE
        tile_width = max(1, canvas_width // width)
        tile_height = max(1, canvas_height // height)
        x_offset = max((canvas_width - (width * tile_width)) // 2, 0)
        y_offset = max((canvas_height - (height * tile_height)) // 2, 0)
        self.board_geometry = (x_offset, y_offset, tile_width, tile_height, width, height)
        images = self.get_scaled_images(tile_width, tile_height)
        self.canvas.delete("all")
        for row_index, row in enumerate(self.maze):
            for column_index, tile in enumerate(row):
                x = x_offset + (column_index * tile_width)
                y = y_offset + (row_index * tile_height)
                self.canvas.create_image(x, y, anchor="nw", image=images[tile])

    def handle_canvas_click(self, event):

        """ Convert a canvas click into an adjacent-cell move. """

        x_offset, y_offset, tile_width, tile_height, width, height = self.board_geometry
        column = (event.x - x_offset) // tile_width
        row = (event.y - y_offset) // tile_height
        if 0 <= row < height and 0 <= column < width:
            self.move_player_to_cell(row, column, source="self")

    def move_player_to_cell(
        self,
        target_row: int,
        target_column: int,
        source: str
    ) -> bool:

        """ Move to a cell only when it is next to the player. """

        row, column = self.player
        row_delta = target_row - row
        column_delta = target_column - column
        if abs(row_delta) + abs(column_delta) != 1:
            return False
        return self.move_player(row_delta, column_delta, source=source)

    def get_scaled_images(
        self,
        tile_width: int,
        tile_height: int
    ) -> dict[str, tk.PhotoImage]:

        """ Return and cache tile images at the requested size. """

        size = (tile_width, tile_height)
        if size not in self.scaled_images:
            if len(self.scaled_images) > 12:
                self.scaled_images.clear()
            self.scaled_images[size] = {
                symbol: self.resize_image(symbol, tile_width, tile_height)
                for symbol in TILE_FILES
            }
        return self.scaled_images[size]

    def resize_image(
        self,
        symbol: str,
        tile_width: int,
        tile_height: int
    ) -> tk.PhotoImage:

        """ Resize one tile with exact scaling or nearest-neighbor sampling. """

        source = self.source_images[symbol]
        source_width = source.width()
        source_height = source.height()
        if tile_width == source_width and tile_height == source_height:
            return source
        if source_width % tile_width == 0 and source_height % tile_height == 0:
            return source.subsample(source_width // tile_width, source_height // tile_height)
        if tile_width % source_width == 0 and tile_height % source_height == 0:
            return source.zoom(tile_width // source_width, tile_height // source_height)

        image = tk.PhotoImage(width=tile_width, height=tile_height)
        source_rows = self.source_pixels[symbol]
        source_transparency = self.source_transparency[symbol]
        source_height = len(source_rows)
        source_width = len(source_rows[0])
        for y in range(tile_height):
            source_y = y * source_height // tile_height
            row = [
                source_rows[source_y][x * source_width // tile_width]
                for x in range(tile_width)
            ]
            image.put("{" + " ".join(row) + "}", to=(0, y))
            for x in range(tile_width):
                source_x = x * source_width // tile_width
                if source_transparency[source_y][source_x]:
                    image.transparency_set(x, y, True)
        return image

    def close_result_window(self):

        """ Close the result dialog if it is open. """

        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        self.result_window = None

    def reload_from_result_window(self):

        """ Close the result dialog and begin a new game. """

        self.close_result_window()
        self.reload_map()

    def write_log(self, message: str = ""):

        """ Append one line to the read-only movement log. """

        self.log_display.configure(state="normal")
        self.log_display.insert("end", f"{message}\n")
        self.log_display.see("end")
        self.log_display.configure(state="disabled")

    def show_result_window(
        self,
        result: str,
        detail_override: str | None = None
    ):

        """ Show a centered win or loss dialog. """

        self.close_result_window()
        title, heading, color, detail = {
            "win": ("ゲームクリア", "クリア", "green", "食べ物を見つけました。"),
            "lose": ("ゲームオーバー", "ゲームオーバー", "red", "罠を踏んでしまいました。"),
        }[result]
        if detail_override is not None:
            detail = detail_override

        window = tk.Toplevel(self)
        self.result_window = window
        window.title(title)
        window.transient(self)
        window.grab_set()
        window.protocol("WM_DELETE_WINDOW", self.close_result_window)

        content = tk.Frame(window, padx=30, pady=24)
        content.pack()
        tk.Label(content, text=heading, fg=color, font=(self.ui_font_family, 32, "bold")).pack()
        tk.Label(content, text=detail, pady=10).pack()

        buttons = tk.Frame(content)
        buttons.pack(pady=(8, 0))
        tk.Button(buttons, text="マップ再読込", command=self.reload_from_result_window).pack(side="left", padx=5)
        tk.Button(buttons, text="閉じる", command=self.close_result_window).pack(side="left", padx=5)

        window.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - window.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - window.winfo_height()) // 2
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        window.focus_set()

    def move_player(
        self,
        row_delta: int,
        column_delta: int,
        source: str = "self"
    ) -> bool:

        """ Apply one move and return whether the turn was accepted. """

        if source == "self" and not self.self_play_enabled.get():
            return False
        if source == "remote" and not self.remote_play_enabled.get():
            return False
        if not self.maze:
            return False
        if self.game_over:
            return False
        row, column = self.player
        target_row, target_column = row + row_delta, column + column_delta
        target_is_in_bounds = (
            0 <= target_row < len(self.maze)
            and 0 <= target_column < len(self.maze[0])
        )
        target = self.maze[target_row][target_column] if target_is_in_bounds else "#"
        hit_wall = target == "#"

        self.no_steps += 1
        self.remaining_energy -= WALL_PENALTY if hit_wall else 1
        if self.no_steps == 1:
            self.write_log()
            self.write_log("開始：")
        if column_delta == 0:
            direction = "上" if row_delta == -1 else "下"
        else:
            direction = "左" if column_delta == -1 else "右"
        self.write_log(
            f"　{self.no_steps}. {direction}、エネルギー{self.remaining_energy}点"
        )
        if self.remaining_energy <= 0:
            self.record_reward("out_of_energy")
            self.status.set("エネルギーを使い切りました。F5またはマップ再読込でもう一度遊べます。")
            self.finish_game(
                "lose",
                "lose: energy left is 0 or lower. Reset with POST /reset.",
                source,
                "エネルギーを使い切りました。",
            )
            self.write_log(f"終了: 敗北 (最終エネルギー = {self.remaining_energy}点)")
            return True
        if hit_wall:
            self.record_reward("wall")
            return False
        self.maze[row][column] = "."
        self.maze[target_row][target_column] = "P"
        self.player = (target_row, target_column)
        self.draw_maze()
        if target == "F":
            self.record_reward("food")
            self.remaining_energy += FOOD_SCORE
            self.status.set("食べ物を見つけました。マップ再読込でもう一度遊べます。")
            self.finish_game(
                "win",
                f"win: food found. Energy left is {self.remaining_energy}. Reset with POST /reset.",
                source,
            )
            self.write_log(f"終了: 勝利 (最終エネルギー = {self.remaining_energy}点)")
            return True
        if target == "T":
            self.record_reward("trap")
            self.remaining_energy -= FOOD_SCORE
            self.status.set("ゲームオーバー。F5またはマップ再読込でもう一度遊べます。")
            self.finish_game(
                "lose",
                f"lose: trap stepped on. Energy left is {self.remaining_energy}. Reset with POST /reset.",
                source,
            )
            self.write_log(f"終了: 敗北 (最終エネルギー = {self.remaining_energy}点)")
            return True
        self.record_reward("step")
        return True


if __name__ == "__main__":
    MazeApp().mainloop()
