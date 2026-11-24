create schema if not exists finagent;
revoke all on schema finagent from public, anon, authenticated;
grant usage on schema finagent to service_role;

create table if not exists finagent.workspace_records (
    kind text not null,
    id text not null,
    owner_id text not null,
    payload jsonb not null,
    updated_at timestamptz not null default now(),
    primary key (kind, id, owner_id)
);

create index if not exists workspace_owner_kind on finagent.workspace_records(owner_id, kind, updated_at desc);
create index if not exists workspace_evidence_document on finagent.workspace_records(owner_id, (payload->>'document_id')) where kind = 'evidence';
create index if not exists workspace_evidence_text on finagent.workspace_records using gin(to_tsvector('english', coalesce(payload->>'text', ''))) where kind = 'evidence';
create index if not exists workspace_job_status on finagent.workspace_records((payload->>'status'), updated_at) where kind = 'jobs';

alter table finagent.workspace_records enable row level security;
drop policy if exists workspace_owner_select on finagent.workspace_records;
create policy workspace_owner_select on finagent.workspace_records for select to authenticated using (owner_id = (select auth.uid())::text);
revoke all on finagent.workspace_records from public, anon, authenticated;
grant select, insert, update, delete on finagent.workspace_records to service_role;

create table if not exists finagent.audit_events (
    entry_index bigint primary key,
    event_id text unique,
    payload jsonb not null
);
alter table finagent.audit_events enable row level security;
revoke all on finagent.audit_events from public, anon, authenticated;
grant select, insert on finagent.audit_events to service_role;

create or replace function finagent.reject_audit_mutation()
returns trigger language plpgsql set search_path = '' as $$
begin
    raise exception 'Audit events are append only';
end;
$$;
revoke execute on function finagent.reject_audit_mutation() from public, anon, authenticated;
drop trigger if exists immutable_audit on finagent.audit_events;
create trigger immutable_audit before update or delete on finagent.audit_events for each row execute function finagent.reject_audit_mutation();

insert into storage.buckets(id, name, public, file_size_limit, allowed_mime_types)
values ('financial-reports', 'financial-reports', false, 20971520, array['application/pdf'])
on conflict(id) do update set public = false, file_size_limit = 20971520, allowed_mime_types = array['application/pdf'];

drop policy if exists financial_reports_owner_read on storage.objects;
create policy financial_reports_owner_read on storage.objects for select to authenticated using (bucket_id = 'financial-reports' and (storage.foldername(name))[1] = (select auth.uid())::text);
