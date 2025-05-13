-- Only create the meetings table if it doesn't exist
CREATE TABLE IF NOT EXISTS meetings (
  id serial primary key,
  user_id uuid not null references auth.users(id),
  meeting_id text not null,
  transcript text,
  context_files text,
  created_at timestamp with time zone default timezone('utc'::text, now()),
  updated_at timestamp with time zone default timezone('utc'::text, now()),
  embeddings vector,
  generated_prompt jsonb,
  chunks text,
  suggestion_count int2 default '0'::smallint
);

-- Add a constraint name for the foreign key (if it doesn't exist)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'meetings_user_id_fkey'
  ) THEN
    ALTER TABLE meetings ADD CONSTRAINT meetings_user_id_fkey 
      FOREIGN KEY (user_id) REFERENCES auth.users(id);
  END IF;
END;
$$;

