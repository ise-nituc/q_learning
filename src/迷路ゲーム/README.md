# Closed Tkinter Maze

Run the application with:

```powershell
python main.py
```

Use the arrow keys to move the player. Press **F5** or **Reload map** after
editing `map.txt`.

The main window has two tabs:

- **Game**: shows the canvas and move log.
- **HTTP settings**: starts/stops API access, sets host/port, and toggles
  self-play and remote-play permissions. When remote play is enabled, an AI
  player can read `GET /state`, move with `POST /move`, and reset a finished
  game with `POST /reset`.

## Map format

The included example is a 10×10 maze. You can change the size by making the
file any rectangular size of at least 3×3. It must be completely closed: every
character on the outside edge must be `#`.

| Symbol | Meaning | Image asset |
| --- | --- | --- |
| `#` | Wall | `Images/wall.png` |
| `.` | Empty path | `Images/none.png` |
| `P` | Player (exactly one) | `Images/play.png` |
| `F` | Food | `Images/food.png` |
| `T` | Trap | `Images/trap.png` |

Do not put spaces in map rows. Blank lines are ignored.

## Export a numbered SVG

Run the following command to recreate `map.txt` as a self-contained SVG with
an orange grid and blue interior-cell numbers with a white contrast shadow
(numbered row-by-row from the
top left, excluding the non-traversable outer border):

```powershell
python map_to_svg.py
```

This creates `map_numbered.svg`. Use `python map_to_svg.py --help` to select a
different map, output path, image directory, or tile size.
