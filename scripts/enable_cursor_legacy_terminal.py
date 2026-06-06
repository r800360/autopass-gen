"""One-shot fix: enable Cursor Legacy Terminal Tool for CARLA localhost RPC."""
import json
import sqlite3
import sys

DB = r"C:\Users\bsach\AppData\Roaming\Cursor\User\globalStorage\state.vscdb"
KEY = "src.vs.platform.reactivestorage.browser.reactiveStorageServiceImpl.persistentStorage.applicationUser"


def main() -> int:
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT value FROM ItemTable WHERE key = ?", (KEY,))
    row = cur.fetchone()
    if not row:
        print("ERROR: applicationUser persistent storage key not found")
        return 1

    data = json.loads(row[0])
    cs = data.setdefault("composerState", {})
    before = cs.get("useLegacyTerminalTool", False)
    if before is True:
        print("useLegacyTerminalTool already enabled")
        return 0

    cs["useLegacyTerminalTool"] = True
    cur.execute("UPDATE ItemTable SET value = ? WHERE key = ?", (json.dumps(data), KEY))
    con.commit()
    con.close()
    print("Updated useLegacyTerminalTool: False -> True")
    print("Restart Cursor (Terminal: Kill All Terminals, then reload window) for this to take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
