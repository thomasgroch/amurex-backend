-- Only create the analytics table if it doesn't exist
CREATE TABLE IF NOT EXISTS analytics (
    id SERIAL PRIMARY KEY,
    uuid TEXT NOT NULL,
    event_type TEXT NOT NULL,
    created_at timestamp with time zone default timezone('utc'::text, now()),
    meeting_id TEXT
);

-- Create indexes for better query performance (if they don't exist)
CREATE INDEX IF NOT EXISTS idx_analytics_uuid ON analytics(uuid);
CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_meeting_id ON analytics(meeting_id);

