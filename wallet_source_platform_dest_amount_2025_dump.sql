select 
source_address as wallet_, 
source_chain, 
case when platform ilike '%debridge%' then 'deBridge' else platform end as platform_,
destination_chain,
sum(amount_usd) as amount_usd,
count(distinct tx_hash) as tx_count
from crosschain.defi.ez_bridge_activity
where block_timestamp >= '2025-01-01'
and block_timestamp < '2025-07-01'
and direction = 'outbound'
and token_is_verified = true
and platform in (
    'across-v3', 'dln_debridge', 'deBridge'
)
and source_chain IN ('base', 'bsc', 'solana', 'arbitrum', 'optimism', 'ethereum', 'polygon',  'zora', 'appchain', 'eclipse', 'soneium', 'abstract')
and destination_chain IN ('base', 'bsc','solana', 'arbitrum', 'optimism', 'ethereum', 'polygon', 'zora', 'appchain', 'eclipse', 'soneium', 'abstract')
group by wallet_, platform_, source_chain, destination_chain