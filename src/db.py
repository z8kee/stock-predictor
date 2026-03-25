import sqlite3
import os
from datetime import datetime

class TradeHistoryDB:
    def __init__(self, db_path='trade_history.db'):
        self.db_path = os.path.join(os.path.dirname(__file__), db_path)
        self.init_db()

    def init_db(self):
        """Initialize the database and create the trade_history table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date_time TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal TEXT NOT NULL CHECK(signal IN ('BUY', 'SELL')),
                    entry_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'SUCCESSFUL', 'FAILED'))
                )
            ''')
            conn.commit()

    def insert_trade(self, ticker, signal, entry_price, target_price, stop_loss, timeframe):
        """Insert a new trade into the database."""
        date_time = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trade_history (date_time, ticker, timeframe, signal, entry_price, target_price, stop_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (date_time, ticker, timeframe, signal, entry_price, target_price, stop_loss))
            conn.commit()
            return cursor.lastrowid

    def update_trade_status(self, trade_id, status):
        """Update the status of a trade."""
        if status not in ['OPEN', 'SUCCESSFUL', 'FAILED']:
            raise ValueError("Status must be 'OPEN', 'SUCCESSFUL', or 'FAILED'")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE trade_history SET status = ? WHERE id = ?
            ''', (status, trade_id))
            conn.commit()

    def get_all_trades(self):
        """Retrieve all trades from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM trade_history ORDER BY date_time DESC')
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    def get_trades_by_ticker(self, ticker):
        """Retrieve trades for a specific ticker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM trade_history WHERE ticker = ? ORDER BY date_time DESC', (ticker,))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

    def get_trades_by_status(self, status):
        """Retrieve trades by status."""
        if status not in ['OPEN', 'SUCCESSFUL', 'FAILED']:
            raise ValueError("Status must be 'OPEN', 'SUCCESSFUL', or 'FAILED'")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM trade_history WHERE status = ? ORDER BY date_time DESC', (status,))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        
    def get_success_ratio(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM trade_history WHERE status IN (?, ?) ORDER BY date_time DESC', ('SUCCESSFUL', 'FAILED'))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]