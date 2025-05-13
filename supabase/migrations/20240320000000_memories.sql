-- Only create the memories table if it doesn't exist
CREATE TABLE IF NOT EXISTS memories (
    id int8 primary key,
    created_at timestamp with time zone default now(),
    user_id uuid,
    content text default ''::text,
    chunks text,
    embeddings vector,
    meeting_id uuid,
    centroid vector
);

-- Create foreign key constraints with explicit names (if they don't exist)
DO $$
BEGIN
  -- Check if the users table exists in public schema and add constraint if it does
  IF EXISTS (
    SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'users'
  ) THEN
    IF NOT EXISTS (
      SELECT 1 FROM pg_constraint WHERE conname = 'memories_user_id_fkey'
    ) THEN
      ALTER TABLE memories
        ADD CONSTRAINT memories_user_id_fkey 
        FOREIGN KEY (user_id) 
        REFERENCES public.users(id);
    END IF;
  END IF;

  -- Check if the late_meeting table exists in public schema and add constraint if it does
  IF EXISTS (
    SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = 'late_meeting'
  ) THEN
    IF NOT EXISTS (
      SELECT 1 FROM pg_constraint WHERE conname = 'memories_meeting_id_fkey'
    ) THEN
      ALTER TABLE memories
        ADD CONSTRAINT memories_meeting_id_fkey 
        FOREIGN KEY (meeting_id) 
        REFERENCES public.late_meeting(id);
    END IF;
  END IF;
END;
$$;

-- Create indexes for foreign keys (if they don't exist)
CREATE INDEX IF NOT EXISTS memories_user_id_idx ON memories(user_id);
CREATE INDEX IF NOT EXISTS memories_meeting_id_idx ON memories(meeting_id);
