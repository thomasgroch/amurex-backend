CREATE TABLE late_meeting (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    meeting_id TEXT NOT NULL,
    user_ids UUID[] NOT NULL,
    meeting_start_time DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(meeting_id)
);

-- Create an index on meeting_id for faster lookups
CREATE INDEX idx_late_meeting_meeting_id ON late_meeting(meeting_id);