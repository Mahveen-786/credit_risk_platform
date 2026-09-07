-- ============================================================================
-- SQL schema reference for the AI-Powered Credit Risk Intelligence Platform.
--
-- Two tables are documented here:
--
-- 1. `application_train` -- the FULL raw Kaggle "Home Credit Default Risk"
--    schema (122 columns), reconstructed from the official competition data
--    dictionary. This is what the raw dataset file actually looks like on
--    disk. Provided so the NL-to-SQL engine (and any evaluator) has a
--    concrete, complete reference to the real dataset structure -- not just
--    the pruned subset the app actually uses.
--
-- 2. `applicants` -- the CURATED, feature-engineered working table the
--    Talk-to-Data engine queries at runtime (see src/talk_to_data/query_runner.py).
--    It is a derived subset of application_train (business-relevant columns
--    only) plus engineered ratio features and post-scoring outputs. It is
--    created dynamically via `con.register("applicants", dataframe)` from
--    the already feature-engineered pandas dataframe (src/data/preprocessor.py
--    + src/ml/predict.py), NOT executed from this DDL -- so schema drift
--    between this file and the running code cannot occur. This file exists
--    for documentation and audit sign-off, reviewable independently of the
--    Python pipeline.
--
-- Column types below are DuckDB-compatible approximations of the real
-- dataset's dtypes for documentation purposes (the Kaggle file itself is a
-- plain CSV with no enforced schema); most columns carry real missingness
-- and should be treated as NULLABLE.
-- ============================================================================


-- ============================================================================
-- 1. application_train -- full raw Kaggle schema (122 columns)
-- ============================================================================
CREATE TABLE IF NOT EXISTS application_train (
    -- Identifiers / target
    SK_ID_CURR                          BIGINT PRIMARY KEY,
    TARGET                               TINYINT,        -- 1 = defaulted, 0 = repaid (train only; absent in application_test)

    -- Loan / contract
    NAME_CONTRACT_TYPE                   VARCHAR,        -- 'Cash loans' | 'Revolving loans'
    AMT_CREDIT                           DOUBLE,
    AMT_ANNUITY                          DOUBLE,
    AMT_GOODS_PRICE                      DOUBLE,
    NAME_TYPE_SUITE                      VARCHAR,        -- who accompanied the applicant

    -- Demographics
    CODE_GENDER                          VARCHAR,        -- 'M' | 'F' | 'XNA'
    FLAG_OWN_CAR                         VARCHAR,        -- 'Y' | 'N'
    FLAG_OWN_REALTY                      VARCHAR,        -- 'Y' | 'N'
    CNT_CHILDREN                         INTEGER,
    CNT_FAM_MEMBERS                      DOUBLE,
    NAME_INCOME_TYPE                     VARCHAR,
    NAME_EDUCATION_TYPE                  VARCHAR,
    NAME_FAMILY_STATUS                   VARCHAR,
    NAME_HOUSING_TYPE                    VARCHAR,
    OCCUPATION_TYPE                      VARCHAR,
    ORGANIZATION_TYPE                    VARCHAR,
    AMT_INCOME_TOTAL                     DOUBLE,
    REGION_POPULATION_RELATIVE           DOUBLE,

    -- Ages / tenure (all stored as negative day-counts relative to application date)
    DAYS_BIRTH                           INTEGER,
    DAYS_EMPLOYED                        INTEGER,        -- contains the documented 365243 anomaly for pensioners/unemployed
    DAYS_REGISTRATION                    DOUBLE,
    DAYS_ID_PUBLISH                      INTEGER,
    DAYS_LAST_PHONE_CHANGE               DOUBLE,
    OWN_CAR_AGE                          DOUBLE,

    -- Contact / device flags
    FLAG_MOBIL                           TINYINT,
    FLAG_EMP_PHONE                       TINYINT,
    FLAG_WORK_PHONE                      TINYINT,
    FLAG_CONT_MOBILE                     TINYINT,
    FLAG_PHONE                           TINYINT,
    FLAG_EMAIL                           TINYINT,

    -- Region / city consistency flags
    REGION_RATING_CLIENT                 TINYINT,
    REGION_RATING_CLIENT_W_CITY          TINYINT,
    REG_REGION_NOT_LIVE_REGION           TINYINT,
    REG_REGION_NOT_WORK_REGION           TINYINT,
    LIVE_REGION_NOT_WORK_REGION          TINYINT,
    REG_CITY_NOT_LIVE_CITY               TINYINT,
    REG_CITY_NOT_WORK_CITY               TINYINT,
    LIVE_CITY_NOT_WORK_CITY              TINYINT,

    -- Application timing
    WEEKDAY_APPR_PROCESS_START           VARCHAR,
    HOUR_APPR_PROCESS_START              TINYINT,

    -- External credit bureau scores
    EXT_SOURCE_1                         DOUBLE,
    EXT_SOURCE_2                         DOUBLE,
    EXT_SOURCE_3                         DOUBLE,

    -- Building/apartment info (AVG variant), where applicant resides
    APARTMENTS_AVG                       DOUBLE,
    BASEMENTAREA_AVG                     DOUBLE,
    YEARS_BEGINEXPLUATATION_AVG          DOUBLE,
    YEARS_BUILD_AVG                      DOUBLE,
    COMMONAREA_AVG                       DOUBLE,
    ELEVATORS_AVG                        DOUBLE,
    ENTRANCES_AVG                        DOUBLE,
    FLOORSMAX_AVG                        DOUBLE,
    FLOORSMIN_AVG                        DOUBLE,
    LANDAREA_AVG                         DOUBLE,
    LIVINGAPARTMENTS_AVG                 DOUBLE,
    LIVINGAREA_AVG                       DOUBLE,
    NONLIVINGAPARTMENTS_AVG              DOUBLE,
    NONLIVINGAREA_AVG                    DOUBLE,

    -- Building/apartment info (MODE variant)
    APARTMENTS_MODE                      DOUBLE,
    BASEMENTAREA_MODE                    DOUBLE,
    YEARS_BEGINEXPLUATATION_MODE         DOUBLE,
    YEARS_BUILD_MODE                     DOUBLE,
    COMMONAREA_MODE                      DOUBLE,
    ELEVATORS_MODE                       DOUBLE,
    ENTRANCES_MODE                       DOUBLE,
    FLOORSMAX_MODE                       DOUBLE,
    FLOORSMIN_MODE                       DOUBLE,
    LANDAREA_MODE                        DOUBLE,
    LIVINGAPARTMENTS_MODE                DOUBLE,
    LIVINGAREA_MODE                      DOUBLE,
    NONLIVINGAPARTMENTS_MODE             DOUBLE,
    NONLIVINGAREA_MODE                   DOUBLE,

    -- Building/apartment info (MEDI variant)
    APARTMENTS_MEDI                      DOUBLE,
    BASEMENTAREA_MEDI                    DOUBLE,
    YEARS_BEGINEXPLUATATION_MEDI         DOUBLE,
    YEARS_BUILD_MEDI                     DOUBLE,
    COMMONAREA_MEDI                      DOUBLE,
    ELEVATORS_MEDI                       DOUBLE,
    ENTRANCES_MEDI                       DOUBLE,
    FLOORSMAX_MEDI                       DOUBLE,
    FLOORSMIN_MEDI                       DOUBLE,
    LANDAREA_MEDI                        DOUBLE,
    LIVINGAPARTMENTS_MEDI                DOUBLE,
    LIVINGAREA_MEDI                      DOUBLE,
    NONLIVINGAPARTMENTS_MEDI             DOUBLE,
    NONLIVINGAREA_MEDI                   DOUBLE,

    -- Building/apartment categorical summaries
    FONDKAPREMONT_MODE                   VARCHAR,
    HOUSETYPE_MODE                       VARCHAR,
    TOTALAREA_MODE                       DOUBLE,
    WALLSMATERIAL_MODE                   VARCHAR,
    EMERGENCYSTATE_MODE                  VARCHAR,

    -- Applicant's social circle default observations
    OBS_30_CNT_SOCIAL_CIRCLE             DOUBLE,
    DEF_30_CNT_SOCIAL_CIRCLE             DOUBLE,
    OBS_60_CNT_SOCIAL_CIRCLE             DOUBLE,
    DEF_60_CNT_SOCIAL_CIRCLE             DOUBLE,

    -- Supporting document flags (20 columns; anonymized document types)
    FLAG_DOCUMENT_2                      TINYINT,
    FLAG_DOCUMENT_3                      TINYINT,
    FLAG_DOCUMENT_4                      TINYINT,
    FLAG_DOCUMENT_5                      TINYINT,
    FLAG_DOCUMENT_6                      TINYINT,
    FLAG_DOCUMENT_7                      TINYINT,
    FLAG_DOCUMENT_8                      TINYINT,
    FLAG_DOCUMENT_9                      TINYINT,
    FLAG_DOCUMENT_10                     TINYINT,
    FLAG_DOCUMENT_11                     TINYINT,
    FLAG_DOCUMENT_12                     TINYINT,
    FLAG_DOCUMENT_13                     TINYINT,
    FLAG_DOCUMENT_14                     TINYINT,
    FLAG_DOCUMENT_15                     TINYINT,
    FLAG_DOCUMENT_16                     TINYINT,
    FLAG_DOCUMENT_17                     TINYINT,
    FLAG_DOCUMENT_18                     TINYINT,
    FLAG_DOCUMENT_19                     TINYINT,
    FLAG_DOCUMENT_20                     TINYINT,
    FLAG_DOCUMENT_21                     TINYINT,

    -- Credit bureau inquiry counts, by lookback window
    AMT_REQ_CREDIT_BUREAU_HOUR           DOUBLE,
    AMT_REQ_CREDIT_BUREAU_DAY            DOUBLE,
    AMT_REQ_CREDIT_BUREAU_WEEK           DOUBLE,
    AMT_REQ_CREDIT_BUREAU_MON            DOUBLE,
    AMT_REQ_CREDIT_BUREAU_QRT            DOUBLE,
    AMT_REQ_CREDIT_BUREAU_YEAR           DOUBLE
);
-- Reference: this reproduces Kaggle's HomeCredit_columns_description.csv
-- structure for application_{train|test}.csv. Cross-check against the
-- official file if column-for-column fidelity is required for audit.


-- ============================================================================
-- 2. applicants -- curated working table (created dynamically at runtime;
--    DDL below is documentation only, see note at top of file)
-- ============================================================================
CREATE TABLE IF NOT EXISTS applicants (
    -- Identifiers / target
    SK_ID_CURR                  BIGINT,
    TARGET                      TINYINT,        -- 1 = defaulted, 0 = repaid

    -- Loan / contract
    NAME_CONTRACT_TYPE          VARCHAR,        -- 'Cash loans' | 'Revolving loans'
    AMT_CREDIT                  DOUBLE,
    AMT_ANNUITY                 DOUBLE,
    AMT_GOODS_PRICE              DOUBLE,

    -- Demographics
    CODE_GENDER                 VARCHAR,
    CNT_CHILDREN                INTEGER,
    CNT_FAM_MEMBERS             INTEGER,
    NAME_FAMILY_STATUS          VARCHAR,
    NAME_HOUSING_TYPE           VARCHAR,
    NAME_EDUCATION_TYPE         VARCHAR,
    AGE_YEARS                   DOUBLE,         -- derived from DAYS_BIRTH

    -- Financials
    AMT_INCOME_TOTAL            DOUBLE,
    NAME_INCOME_TYPE            VARCHAR,
    OCCUPATION_TYPE              VARCHAR,
    EMPLOYED_YEARS               DOUBLE,        -- derived from DAYS_EMPLOYED
    EMPLOYED_ANOMALY_FLAG        TINYINT,       -- 1 if DAYS_EMPLOYED == 365243

    -- Credit bureau signal
    EXT_SOURCE_1                 DOUBLE,
    EXT_SOURCE_2                 DOUBLE,
    EXT_SOURCE_3                 DOUBLE,
    EXT_SOURCE_MEAN               DOUBLE,       -- derived
    EXT_SOURCE_MISSING_COUNT      TINYINT,      -- derived
    AMT_REQ_CREDIT_BUREAU_QRT     INTEGER,

    -- Derived risk ratios
    ANNUITY_TO_INCOME             DOUBLE,
    CREDIT_TO_INCOME              DOUBLE,
    CREDIT_TO_GOODS_RATIO         DOUBLE,
    INCOME_PER_FAM_MEMBER         DOUBLE,

    -- Scoring outputs (added post-inference, see src/ml/predict.py)
    PD_SCORE                      DOUBLE,       -- probability of default, 0-1
    RISK_BAND                     VARCHAR,      -- 'Low' | 'Medium' | 'High'
    AI_CREDIT_SCORE                INTEGER,     -- 300-850 presentation score
    EXPECTED_LOSS                  DOUBLE       -- PD x LGD x AMT_CREDIT
);

-- Example audit queries (also available as the verified offline library in
-- src/talk_to_data/query_runner.py::OFFLINE_QUERY_LIBRARY):

-- Overall default rate
-- SELECT ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct FROM applicants;

-- Risk band distribution vs. actual outcomes
-- SELECT RISK_BAND, COUNT(*) AS n, ROUND(AVG(TARGET)*100,2) AS actual_default_rate_pct
-- FROM applicants GROUP BY RISK_BAND ORDER BY actual_default_rate_pct DESC;
