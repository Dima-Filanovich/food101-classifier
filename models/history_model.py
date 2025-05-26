from .database import get_connection

def add_history(user_id, image_name, prediction, confidence):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (user_id, image_name, prediction, confidence)
        VALUES (?, ?, ?, ?)
    """, (user_id, image_name, prediction, float(confidence)))
    conn.commit()
    conn.close()

def get_history_by_user(user_id, limit=10):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT image_name, prediction, confidence, timestamp
        FROM history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, limit))
    records = cursor.fetchall()
    conn.close()
    return records
