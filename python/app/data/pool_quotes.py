from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional

from web3 import Web3

# Default Uniswap v3 Quoter (Ethereum mainnet). Override per-chain via config.
QUOTER_V3 = "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"

ABI_QUOTER = [
    {
        "inputs": [
            {"internalType": "address", "name": "tokenIn", "type": "address"},
            {"internalType": "address", "name": "tokenOut", "type": "address"},
            {"internalType": "uint24", "name": "fee", "type": "uint24"},
            {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
            {"internalType": "uint160", "name": "sqrtPriceLimitX96", "type": "uint160"},
        ],
        "name": "quoteExactInputSingle",
        "outputs": [
            {"internalType": "uint256", "name": "amountOut", "type": "uint256"},
        ],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

UNISWAP_V3_POOL_ABI = [
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},
            {"internalType":"int24","name":"tick","type":"int24"},
            {"internalType":"uint16","name":"observationIndex","type":"uint16"},
            {"internalType":"uint16","name":"observationCardinality","type":"uint16"},
            {"internalType":"uint16","name":"observationCardinalityNext","type":"uint16"},
            {"internalType":"uint8","name":"feeProtocol","type":"uint8"},
            {"internalType":"bool","name":"unlocked","type":"bool"}
        ],
        "stateMutability":"view",
        "type":"function"
    }
]


def v3_mid_price(provider: Web3, pool_address: str, dec0: int, dec1: int) -> float:
    pool = provider.eth.contract(
        address=Web3.to_checksum_address(pool_address),
        abi=UNISWAP_V3_POOL_ABI,
    )
    sqrtPriceX96 = pool.functions.slot0().call()[0]
    price = (sqrtPriceX96 / (2**96)) ** 2
    price_adj = price * (10**dec0) / (10**dec1)
    return float(price_adj)


def get_uniswap_v3_pool_quote(
    provider: Web3,
    pool_address: str,
    token_in: str,
    token_out: str,
    fee: int,               # e.g. 500 (0.05%), 3000 (0.3%)
    amount_in_wei: int,
    token_in_decimals: int,
    token_out_decimals: int,
    quoter_address: Optional[str] = None,
) -> Dict[str, float]:
    # 1) mid price from slot0
    mid_price = v3_mid_price(provider, pool_address, token_in_decimals, token_out_decimals)

    # 2) executable quote from QuoterV3
    quoter = provider.eth.contract(
        address=Web3.to_checksum_address(quoter_address or QUOTER_V3),
        abi=ABI_QUOTER,
    )
    amount_out_wei = quoter.functions.quoteExactInputSingle(
        Web3.to_checksum_address(token_in),
        Web3.to_checksum_address(token_out),
        fee,
        amount_in_wei,
        0,  # sqrtPriceLimitX96 = 0 (no limit)
    ).call()

    amount_in = amount_in_wei / (10 ** token_in_decimals)
    amount_out = amount_out_wei / (10 ** token_out_decimals)
    exec_price = amount_out / amount_in if amount_in > 0 else 0.0

    if mid_price and mid_price > 0:
        impact_bps = (exec_price / mid_price - 1.0) * 10_000
    else:
        impact_bps = 0.0

    return {
        "mid_price": mid_price,
        "exec_price": exec_price,
        "amount_in": amount_in,
        "amount_out": amount_out,
        "price_impact_bps": impact_bps,
        "lp_fee_bps": fee / 100.0,
    }
    
    
def fetch_pool_quote(
    coin: str,
    pool_address: str,
    symbol: str,
    size_usd: float,
    indicator_row: Optional[Any] = None,
    risk_score: Optional[float] = None,
    fee_bps_default: int = 30,      # in *bps*: 5, 30, etc.
    chain: str = "polygon",
    timeout: int = 10,
    rpc_url: Optional[str] = None,
    token_in: Optional[str] = None,
    token_out: Optional[str] = None,
    token_in_decimals: int = 18,
    token_out_decimals: int = 6,
    sample_amount_in: float = 0.01,  # in *token_in* units, e.g. 0.01 WETH
    quoter_address: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return a lightweight execution quote snapshot for the requested pool.
    Uses Uniswap v3 quote when rpc_url + token addresses are provided.
    """
    size_usd = float(max(size_usd, 0.0))

    reference_price = None
    if indicator_row is not None:
        try:
            reference_price = float(indicator_row.get("Close"))
        except Exception:
            reference_price = None

    # --- Uniswap mid & exec price ---
    base_price = None
    provider_url = rpc_url or os.environ.get("POLYGON_RPC_URL") or os.environ.get("ETH_RPC_URL")
    if provider_url and token_in and token_out:
        try:
            w3 = Web3(Web3.HTTPProvider(provider_url))
            if w3.is_connected():
                # sample amount for price/impact
                amount_in_wei = int(sample_amount_in * 10**token_in_decimals)
                uni_quote = get_uniswap_v3_pool_quote(
                    provider=w3,
                    pool_address=pool_address,
                    token_in=token_in,
                    token_out=token_out,
                    fee=int(fee_bps_default * 100),  # 5 bps -> 500 fee tier
                    amount_in_wei=amount_in_wei,
                    token_in_decimals=token_in_decimals,
                    token_out_decimals=token_out_decimals,
                    quoter_address=quoter_address,
                )
                # prefer executable price; fall back to mid
                base_price = uni_quote.get("exec_price") or uni_quote.get("mid_price")
        except Exception:
            base_price = None

    if base_price is None:
        base_price = reference_price or 0.0

    # --- volatility / risk inputs ---
    vol_z = None
    if indicator_row is not None:
        try:
            vol_z = float(indicator_row.get("vol_z"))
        except Exception:
            vol_z = None

    magnitude = abs(vol_z) if vol_z is not None and not math.isnan(vol_z) else 1.0
    risk = float(risk_score) if risk_score is not None else 0.5

    # slippage heuristic grows with risk & size
    slip_base = 5 + 10 * risk
    slip_factor = 1 + magnitude * 0.2
    size_factor = 1 + max(0.0, size_usd - 1000) / 10000
    entry_slip_bps = slip_base * slip_factor * size_factor
    exit_slip_bps = entry_slip_bps * 0.8

    # gas assumptions (USD) – you can later replace this with real estimateGas
    gas_open_usd = 0.8 + 1.5 * risk
    gas_close_usd = gas_open_usd

    mev_buffer_bps = 5 + 10 * risk

    basis_bps = None
    if reference_price not in (None, 0) and base_price not in (None, 0):
        try:
            basis_bps = ((base_price - reference_price) / reference_price) * 10000
        except Exception:
            basis_bps = None

    quote = {
        "coin": coin,
        "pool_address": pool_address,
        "chain": chain,
        "symbol": symbol,
        "mid_price": base_price,
        "reference_price": reference_price,
        "basis_bps": basis_bps,
        "size_usd": size_usd,
        "fee_bps": float(fee_bps_default),
        "entry_slippage_bps": entry_slip_bps,
        "exit_slippage_bps": exit_slip_bps,
        "gas_open_usd": gas_open_usd,
        "gas_close_usd": gas_close_usd,
        "mev_buffer_bps": mev_buffer_bps,
        "borrow_bps": 0.0,
        "timestamp": time.time(),
        "inputs": {
            "vol_z": vol_z,
            "risk_score": risk,
        },
    }
    return quote
