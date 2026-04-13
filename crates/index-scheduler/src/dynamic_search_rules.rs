use std::collections::BTreeSet;
use std::env::VarError;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use bumpalo::Bump;
use meilisearch_types::dynamic_search_rules::{Condition, DynamicSearchRule, RuleUid};
use meilisearch_types::heed;
use meilisearch_types::heed::{CompactionOption, EnvFlags, RwTxn, WithoutTls};
use meilisearch_types::milli::documents::documents_batch_reader_from_objects;
use meilisearch_types::milli::index::PrefixSearch;
use meilisearch_types::milli::progress::Progress;
use meilisearch_types::milli::update::new::indexer;
use meilisearch_types::milli::update::{
    ClearDocuments, IndexDocuments, IndexDocumentsConfig, IndexDocumentsMethod, IndexerConfig,
};
use meilisearch_types::milli::{
    self, all_obkv_to_json, filtered_universe, CreateOrOpen, Filter, FilterableAttributesRule,
    Index,
};
use roaring::RoaringBitmap;
use serde_json::{json, Value};
use time::OffsetDateTime;

use crate::Result;
use crate::{Error, IndexSchedulerOptions};

const NUMBER_OF_DATABASES: u32 = 0;
const HIDDEN_INDEX_UID: &str = "__dynamic-search-rules";
pub(crate) const INDEX_DIRECTORY_NAME: &str = "dynamic-search-rules-index";
const MAX_HIDDEN_INDEX_READERS: u32 = 128;

const FIELD_QUERY_CONTAINS: &str = "__dsr.query.contains";
const FIELD_QUERY_EMPTY_STATE: &str = "__dsr.query.emptyState";
const FIELD_TIME_START: &str = "__dsr.time.start";
const FIELD_TIME_END: &str = "__dsr.time.end";
const FIELD_SELECTOR_INDEX_UID: &str = "__dsr.action.selector.indexUid";

const QUERY_EMPTY: &str = "empty";
const QUERY_NON_EMPTY: &str = "nonEmpty";
const GLOBAL_SELECTOR_INDEX_UID: &str = "$global";

#[derive(Clone)]
pub(crate) struct DynamicSearchRulesStore {
    index_path: PathBuf,
    enable_mdb_writemap: bool,
    map_size: usize,
    index: Arc<Mutex<Option<Index>>>,
    indexer_config: Arc<IndexerConfig>,
    ip_policy: http_client::policy::IpPolicy,
}

impl DynamicSearchRulesStore {
    pub(crate) const fn nb_db() -> u32 {
        NUMBER_OF_DATABASES
    }

    pub fn new(
        _env: &heed::Env<WithoutTls>,
        _wtxn: &mut RwTxn<'_>,
        options: &IndexSchedulerOptions,
        index_map_size: usize,
    ) -> Result<Self> {
        let index_path = index_path(&options.indexes_path);
        std::fs::create_dir_all(&index_path)?;

        Ok(Self {
            index_path,
            enable_mdb_writemap: options.enable_mdb_writemap,
            map_size: index_map_size,
            index: Arc::new(Mutex::new(None)),
            indexer_config: options.indexer_config.clone(),
            ip_policy: options.ip_policy.clone(),
        })
    }

    pub fn replace_all(&self, rules: impl IntoIterator<Item = DynamicSearchRule>) -> Result<()> {
        let index = self.index()?;
        let mut wtxn = index.write_txn()?;
        ClearDocuments::new(&mut wtxn, &index).execute().map_err(error)?;
        let objects = rules
            .into_iter()
            .map(|rule| rule_to_object(rule, &index))
            .collect::<Result<Vec<_>, _>>()?;
        if !objects.is_empty() {
            let reader = documents_batch_reader_from_objects(objects);
            let embedder_stats: Arc<milli::progress::EmbedderStats> = Default::default();
            let builder = IndexDocuments::new(
                &mut wtxn,
                &index,
                &self.indexer_config,
                IndexDocumentsConfig {
                    update_method: IndexDocumentsMethod::ReplaceDocuments,
                    ..Default::default()
                },
                |_indexing_step| {},
                || false,
                &embedder_stats,
                &self.ip_policy,
            )
            .map_err(error)?;

            let (builder, user_result) = builder.add_documents(reader).map_err(error)?;
            user_result.map_err(|error| {
                Error::from_milli(error.into(), Some(HIDDEN_INDEX_UID.to_owned()))
            })?;
            builder.execute().map_err(error)?;
        }
        wtxn.commit().map_err(Into::into)
    }

    pub fn all_rules(&self) -> Result<Vec<DynamicSearchRule>> {
        let index = self.index()?;
        let rtxn = index.read_txn()?;
        let docids = index.documents_ids(&rtxn)?;
        let mut rules = rules_by_docids(&index, &rtxn, docids)?;
        rules.sort_by(|lhs, rhs| lhs.uid.cmp(&rhs.uid));
        Ok(rules)
    }

    pub fn get_rule(&self, uid: &RuleUid) -> Result<Option<DynamicSearchRule>> {
        let index = self.index()?;
        let rtxn = index.read_txn()?;
        let Some(docid) = index.external_documents_ids().get(&rtxn, uid.as_str())? else {
            return Ok(None);
        };

        let mut rules = rules_by_docids(&index, &rtxn, std::iter::once(docid))?;
        Ok(rules.pop())
    }

    pub fn put_rule(&self, rule: &DynamicSearchRule) -> Result<()> {
        let index = self.index()?;
        let mut wtxn = index.write_txn()?;
        let reader =
            documents_batch_reader_from_objects(vec![rule_to_object(rule.clone(), &index)?]);
        let embedder_stats: Arc<milli::progress::EmbedderStats> = Default::default();
        let builder = IndexDocuments::new(
            &mut wtxn,
            &index,
            &self.indexer_config,
            IndexDocumentsConfig {
                update_method: IndexDocumentsMethod::ReplaceDocuments,
                ..Default::default()
            },
            |_indexing_step| {},
            || false,
            &embedder_stats,
            &self.ip_policy,
        )
        .map_err(error)?;

        let (builder, user_result) = builder.add_documents(reader).map_err(error)?;
        user_result
            .map_err(|error| Error::from_milli(error.into(), Some(HIDDEN_INDEX_UID.to_owned())))?;
        builder.execute().map_err(error)?;
        wtxn.commit().map_err(Into::into)
    }

    pub fn delete_rule(&self, uid: &RuleUid) -> Result<bool> {
        let index = self.index()?;
        let mut wtxn = index.write_txn()?;
        if index.external_documents_ids().get(&wtxn, uid.as_str())?.is_none() {
            return Ok(false);
        }

        delete_documents(
            &index,
            &self.indexer_config,
            &self.ip_policy,
            &mut wtxn,
            &[uid.as_str()],
        )?;
        wtxn.commit()?;
        Ok(true)
    }

    pub fn candidate_rules(
        &self,
        query: Option<&str>,
        index_uid: &str,
        now: OffsetDateTime,
        progress: &Progress,
    ) -> Result<Vec<DynamicSearchRule>> {
        let index = self.index()?;
        let rtxn = index.read_txn()?;
        let query = query.and_then(|query| {
            let trimmed = query.trim();
            (!trimmed.is_empty()).then_some(trimmed)
        });

        let docids = match query {
            Some(query) => query_candidates(&index, &rtxn, query, index_uid, now, progress)?,
            None => placeholder_candidates(&index, &rtxn, index_uid, now, progress)?,
        };

        if docids.is_empty() {
            return Ok(Vec::new());
        }

        let mut rules = rules_by_docids(&index, &rtxn, docids)?;
        rules.sort_by(|lhs, rhs| lhs.uid.cmp(&rhs.uid));
        Ok(rules)
    }

    pub fn copy_to_snapshot(&self, snapshot_root: &Path, option: CompactionOption) -> Result<()> {
        let index = self.index()?;
        let dst = snapshot_root.join("indexes").join(INDEX_DIRECTORY_NAME);
        std::fs::create_dir_all(&dst)?;
        index.copy_to_path(dst.join("data.mdb"), option).map_err(error)?;
        Ok(())
    }

    fn index(&self) -> Result<Index> {
        let mut index = self.index.lock().unwrap();
        if let Some(index) = index.as_ref() {
            return Ok(index.clone());
        }

        let opened =
            open_index(&self.index_path, self.enable_mdb_writemap, self.map_size).map_err(error)?;
        ensure_settings(&opened, &self.indexer_config, &self.ip_policy)?;

        *index = Some(opened.clone());
        Ok(opened)
    }
}

fn searchable_fields() -> Vec<String> {
    vec![FIELD_QUERY_CONTAINS.to_string()]
}

fn filterable_fields() -> Vec<FilterableAttributesRule> {
    vec![
        FilterableAttributesRule::Field("active".to_string()),
        FilterableAttributesRule::Field(FIELD_QUERY_CONTAINS.to_string()),
        FilterableAttributesRule::Field(FIELD_QUERY_EMPTY_STATE.to_string()),
        FilterableAttributesRule::Field(FIELD_TIME_START.to_string()),
        FilterableAttributesRule::Field(FIELD_TIME_END.to_string()),
        FilterableAttributesRule::Field(FIELD_SELECTOR_INDEX_UID.to_string()),
    ]
}

fn ensure_settings(
    index: &Index,
    indexer_config: &IndexerConfig,
    ip_policy: &http_client::policy::IpPolicy,
) -> Result<()> {
    let searchable_fields = searchable_fields();
    let searchable_field_refs = searchable_fields.iter().map(String::as_str).collect::<Vec<_>>();
    let filterable_fields = filterable_fields();

    let rtxn = index.read_txn()?;
    let needs_primary_key = index.primary_key(&rtxn)? != Some("uid");
    let needs_searchable_fields =
        index.user_defined_searchable_fields(&rtxn)? != Some(searchable_field_refs);
    let needs_filterable_fields = index.filterable_attributes_rules(&rtxn)? != filterable_fields;
    let needs_typos = index.authorize_typos(&rtxn)?;
    let needs_prefix_search =
        index.prefix_search(&rtxn)?.unwrap_or_default() != PrefixSearch::Disabled;
    drop(rtxn);

    if !(needs_primary_key
        || needs_searchable_fields
        || needs_filterable_fields
        || needs_typos
        || needs_prefix_search)
    {
        return Ok(());
    }

    let mut wtxn = index.write_txn()?;
    let mut settings = milli::update::Settings::new(&mut wtxn, index, indexer_config);
    if needs_primary_key {
        settings.set_primary_key("uid".to_string());
    }
    if needs_searchable_fields {
        settings.set_searchable_fields(searchable_fields);
    }
    if needs_filterable_fields {
        settings.set_filterable_fields(filterable_fields);
    }
    if needs_typos {
        settings.set_authorize_typos(false);
    }
    if needs_prefix_search {
        settings.set_prefix_search(PrefixSearch::Disabled);
    }
    settings
        .execute(&|| false, &Progress::default(), ip_policy, Default::default())
        .map_err(error)?;
    wtxn.commit().map_err(Into::into)
}

fn query_candidates(
    index: &Index,
    rtxn: &heed::RoTxn<'_>,
    query: &str,
    index_uid: &str,
    now: OffsetDateTime,
    progress: &Progress,
) -> Result<RoaringBitmap> {
    let query_filter = non_empty_query_filter(index_uid, now);
    let _ = query;
    filtered_candidates(index, rtxn, Some(&query_filter), progress)
}

fn placeholder_candidates(
    index: &Index,
    rtxn: &heed::RoTxn<'_>,
    index_uid: &str,
    now: OffsetDateTime,
    progress: &Progress,
) -> Result<RoaringBitmap> {
    filtered_candidates(index, rtxn, Some(&placeholder_query_filter(index_uid, now)), progress)
}

fn filtered_candidates(
    index: &Index,
    rtxn: &heed::RoTxn<'_>,
    filter: Option<&str>,
    progress: &Progress,
) -> Result<RoaringBitmap> {
    let filter = match filter {
        Some(filter) => {
            let filter = Filter::from_str(filter)
                .map_err(error)?
                .ok_or_else(|| Error::CorruptedTaskQueue)?;
            let mut filters =
                crate::filter::filters_into_index_filters_unchecked(vec![Some(filter)])?;
            filters.pop().flatten()
        }
        None => None,
    };

    filtered_universe(index, rtxn, &filter, progress).map_err(error)
}

fn base_filter(index_uid: &str, now: OffsetDateTime) -> String {
    let now = now.unix_timestamp();
    format!(
        "active = true AND ({FIELD_TIME_START} <= {now} OR {FIELD_TIME_START} NOT EXISTS) AND ({FIELD_TIME_END} >= {now} OR {FIELD_TIME_END} NOT EXISTS) AND ({FIELD_SELECTOR_INDEX_UID} = {} OR {FIELD_SELECTOR_INDEX_UID} = {})",
        filter_value(index_uid),
        filter_value(GLOBAL_SELECTOR_INDEX_UID),
    )
}

fn non_empty_query_filter(index_uid: &str, now: OffsetDateTime) -> String {
    format!(
        "{} AND ({FIELD_QUERY_EMPTY_STATE} = {} OR {FIELD_QUERY_EMPTY_STATE} NOT EXISTS)",
        base_filter(index_uid, now),
        filter_value(QUERY_NON_EMPTY),
    )
}

fn placeholder_query_filter(index_uid: &str, now: OffsetDateTime) -> String {
    format!(
        "{} AND ({FIELD_QUERY_EMPTY_STATE} = {} OR {FIELD_QUERY_EMPTY_STATE} NOT EXISTS) AND {FIELD_QUERY_CONTAINS} NOT EXISTS",
        base_filter(index_uid, now),
        filter_value(QUERY_EMPTY),
    )
}

fn filter_value(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a filter value cannot fail")
}

fn delete_documents(
    index: &Index,
    indexer_config: &IndexerConfig,
    ip_policy: &http_client::policy::IpPolicy,
    wtxn: &mut heed::RwTxn<'_>,
    external_ids: &[&str],
) -> Result<()> {
    let rtxn = index.read_txn()?;
    let db_fields_ids_map = index.fields_ids_map(&rtxn)?;
    let mut new_fields_ids_map = db_fields_ids_map.clone();

    let mut operations = indexer::IndexOperations::new();
    operations.delete_documents(external_ids);

    let indexer_alloc = Bump::new();
    let (document_changes, operation_stats, primary_key) = operations
        .into_changes(
            &indexer_alloc,
            index,
            &rtxn,
            None,
            &mut new_fields_ids_map,
            &|| false,
            Progress::default(),
            None,
        )
        .map_err(error)?;

    if let Some(error) = operation_stats.into_iter().find_map(|stat| stat.error) {
        return Err(Error::from_milli(error.into(), Some(HIDDEN_INDEX_UID.to_owned())));
    }

    indexer_config
        .thread_pool
        .install(|| {
            indexer::index(
                wtxn,
                index,
                &indexer_config.thread_pool,
                indexer_config.grenad_parameters(),
                &db_fields_ids_map,
                new_fields_ids_map,
                primary_key,
                &document_changes,
                Default::default(),
                &|| false,
                &Progress::default(),
                ip_policy,
                &Default::default(),
            )
        })
        .map_err(|error| Error::ProcessBatchPanicked(error.to_string()))?
        .map_err(error)?;

    Ok(())
}

fn rules_by_docids(
    index: &Index,
    rtxn: &heed::RoTxn<'_>,
    docids: impl IntoIterator<Item = milli::DocumentId>,
) -> Result<Vec<DynamicSearchRule>> {
    let fields_ids_map = index.fields_ids_map(rtxn)?;
    index
        .documents(rtxn, docids)
        .map_err(error)?
        .into_iter()
        .map(|(_docid, document)| {
            let json = all_obkv_to_json(document, &fields_ids_map).map_err(error)?;
            serde_json::from_value(Value::Object(json)).map_err(serde_error)
        })
        .collect()
}

fn index_path(indexes_path: &Path) -> PathBuf {
    indexes_path.join(INDEX_DIRECTORY_NAME)
}

fn open_index(path: &Path, enable_mdb_writemap: bool, map_size: usize) -> milli::Result<Index> {
    let options = heed::EnvOpenOptions::new();
    let mut options = options.read_txn_without_tls();
    options.map_size(crate::utils::clamp_to_page_size(map_size));

    let max_readers = match std::env::var("MEILI_EXPERIMENTAL_INDEX_MAX_READERS") {
        Ok(value) => u32::from_str(&value).unwrap().min(MAX_HIDDEN_INDEX_READERS),
        Err(VarError::NotPresent) => MAX_HIDDEN_INDEX_READERS,
        Err(VarError::NotUnicode(value)) => panic!(
            "Invalid unicode for the `MEILI_EXPERIMENTAL_INDEX_MAX_READERS` env var: {value:?}"
        ),
    };
    options.max_readers(max_readers);
    if enable_mdb_writemap {
        unsafe { options.flags(EnvFlags::WRITE_MAP) };
    }

    let create_or_open = if path.join("data.mdb").exists() {
        CreateOrOpen::Open
    } else {
        CreateOrOpen::create_without_shards()
    };
    let now = OffsetDateTime::now_utc();

    Index::new_with_creation_dates(options, path, now, now, create_or_open)
}

fn rule_to_object(rule: DynamicSearchRule, _index: &Index) -> Result<milli::Object> {
    let mut object = match serde_json::to_value(&rule).map_err(serde_error)? {
        Value::Object(object) => object,
        _ => unreachable!("dynamic search rules documents always serialize as objects"),
    };

    let mut query_contains = BTreeSet::new();
    let mut query_empty_state = BTreeSet::new();
    let mut selector_index_uids = BTreeSet::new();
    let mut timestamp_start = None;
    let mut timestamp_end = None;

    for condition in &rule.conditions {
        match condition {
            Condition::Query { is_empty, contains } => {
                match is_empty {
                    Some(true) => {
                        query_empty_state.insert(QUERY_EMPTY.to_string());
                    }
                    Some(false) => {
                        query_empty_state.insert(QUERY_NON_EMPTY.to_string());
                    }
                    None => {}
                }

                if let Some(contains) = contains {
                    query_contains.insert(milli::normalize_facet(contains));
                }
            }
            Condition::Time { start, end } => {
                if let Some(start) = start {
                    let start = start.unix_timestamp();
                    timestamp_start =
                        Some(timestamp_start.map_or(start, |current: i64| current.max(start)));
                }
                if let Some(end) = end {
                    let end = end.unix_timestamp();
                    timestamp_end =
                        Some(timestamp_end.map_or(end, |current: i64| current.min(end)));
                }
            }
        }
    }

    for action in &rule.actions {
        match action.selector.index_uid.as_ref() {
            Some(index_uid) => {
                selector_index_uids.insert(index_uid.to_string());
            }
            None => {
                selector_index_uids.insert(GLOBAL_SELECTOR_INDEX_UID.to_string());
            }
        }
    }

    if !query_contains.is_empty() {
        object.insert(FIELD_QUERY_CONTAINS.to_string(), json!(query_contains));
    }
    if !query_empty_state.is_empty() {
        object.insert(FIELD_QUERY_EMPTY_STATE.to_string(), json!(query_empty_state));
    }
    if let Some(timestamp_start) = timestamp_start {
        object.insert(FIELD_TIME_START.to_string(), json!(timestamp_start));
    }
    if let Some(timestamp_end) = timestamp_end {
        object.insert(FIELD_TIME_END.to_string(), json!(timestamp_end));
    }
    if !selector_index_uids.is_empty() {
        object.insert(FIELD_SELECTOR_INDEX_UID.to_string(), json!(selector_index_uids));
    }

    Ok(object)
}

fn error(error: milli::Error) -> Error {
    Error::from_milli(error, Some(HIDDEN_INDEX_UID.to_owned()))
}

fn serde_error(error: serde_json::Error) -> Error {
    Error::from_milli(
        milli::InternalError::SerdeJson(error).into(),
        Some(HIDDEN_INDEX_UID.to_owned()),
    )
}
