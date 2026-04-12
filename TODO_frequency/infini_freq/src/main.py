import argparse
import csv
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
	from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
	tqdm = None

API_URL_DEFAULT = "https://api.infini-gram.io/"
INDEX_DEFAULT = "v4_olmo-mix-1124_llama"


logger = logging.getLogger("infini_freq")


def _configure_logging(level: str) -> None:
	if logger.handlers:
		return
	logging.basicConfig(
		level=getattr(logging, level.upper(), logging.INFO),
		format="%(asctime)s %(levelname)s %(message)s",
	)


@dataclass(frozen=True)
class CountResult:
	query: str
	index: str
	count: Optional[int]
	approx: Optional[bool]
	latency: Optional[float]
	token_ids: Optional[List[int]]
	tokens: Optional[List[str]]
	error: Optional[str]


class InfiniGramClient:
	def __init__(
		self,
		api_url: str = API_URL_DEFAULT,
		index: str = INDEX_DEFAULT,
		timeout_s: float = 30.0,
		retries: int = 6,
		backoff_s: float = 0.5,
	) -> None:
		self.api_url = api_url
		self.index = index
		self.timeout_s = timeout_s
		self.retries = retries
		self.backoff_s = backoff_s

	def count(self, query: str) -> CountResult:
		payload = {
			"index": self.index,
			"query_type": "count",
			"query": query,
		}
		response = self._post_json(payload)
		if response is None:
			return CountResult(
				query=query,
				index=self.index,
				count=None,
				approx=None,
				latency=None,
				token_ids=None,
				tokens=None,
				error="request_failed",
			)

		if isinstance(response, dict) and "error" in response:
			return CountResult(
				query=query,
				index=self.index,
				count=None,
				approx=None,
				latency=response.get("latency"),
				token_ids=response.get("token_ids"),
				tokens=response.get("tokens"),
				error=str(response.get("error")),
			)

		return CountResult(
			query=query,
			index=self.index,
			count=response.get("count"),
			approx=response.get("approx"),
			latency=response.get("latency"),
			token_ids=response.get("token_ids"),
			tokens=response.get("tokens"),
			error=None,
		)

	def _post_json(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		data = json.dumps(payload).encode("utf-8")
		req = urllib.request.Request(
			self.api_url,
			data=data,
			headers={"Content-Type": "application/json"},
			method="POST",
		)

		last_err: Optional[str] = None
		for attempt in range(self.retries + 1):
			try:
				with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
					raw = resp.read().decode("utf-8")
				return json.loads(raw)
			except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
				last_err = f"{type(e).__name__}: {e}"
				if attempt >= self.retries:
					break
				time.sleep(self.backoff_s * (2**attempt))

		return {"error": last_err or "request_failed"}


def _read_idioms(input_path: Path, column: Optional[str]) -> Tuple[List[Dict[str, Any]], str]:
	"""Returns (rows, idiom_column_used).

	If input is CSV, returns rows as dicts.
	If input is a text file, returns rows like {"idiom": <line>} and idiom_column_used="idiom".
	"""
	suffix = input_path.suffix.lower()
	if suffix in {".csv", ".tsv"}:
		dialect: csv.Dialect = csv.excel
		if suffix == ".tsv":
			dialect = csv.excel_tab

		with input_path.open("r", encoding="utf-8", newline="") as f:
			reader = csv.DictReader(f, dialect=dialect)
			rows = list(reader)

		if not rows:
			raise ValueError(f"No rows found in {input_path}")

		fieldnames = list(rows[0].keys())
		if column is None:
			for candidate in ("base_form", "extracted_idiom", "idiom", "phrase"):
				if candidate in fieldnames:
					column = candidate
					break
		if column is None or column not in fieldnames:
			raise ValueError(
				f"Could not determine idiom column. Available columns: {fieldnames}. "
				f"Pass --column explicitly."
			)
		return rows, column

	# Treat as newline-separated idioms.
	idioms: List[str] = []
	with input_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			idioms.append(line)

	rows = [{"idiom": x} for x in idioms]
	return rows, "idiom"


def _json_dumps_compact(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def run(
	input_path: Path,
	output_path: Path,
	column: Optional[str],
	index: str,
	api_url: str,
	timeout_s: float,
	retries: int,
	sleep_s: float,
	disable_tqdm: bool,
) -> None:
	rows, idiom_col = _read_idioms(input_path, column)
	client = InfiniGramClient(api_url=api_url, index=index, timeout_s=timeout_s, retries=retries)

	logger.info(
		"Querying infini-gram counts: n_rows=%d index=%s column=%s api=%s",
		len(rows),
		index,
		idiom_col,
		api_url,
	)

	# Total tokens for normalization (Infini-gram: empty string returns corpus token count).
	total_tokens_result = client.count("")
	total_tokens: Optional[int] = total_tokens_result.count
	if total_tokens is None:
		logger.warning("Failed to fetch total token count (normalization disabled): %s", total_tokens_result.error)
	else:
		logger.info("Fetched corpus total tokens: %d", total_tokens)

	if tqdm is None and not disable_tqdm:
		logger.warning("tqdm is not installed; progress bar disabled. Install with: pip install tqdm")

	out_rows: List[Dict[str, Any]] = []
	n_errors = 0
	iter_rows = enumerate(rows)
	if tqdm is not None and not disable_tqdm:
		iter_rows = tqdm(iter_rows, total=len(rows), unit="idiom")

	for i, row in iter_rows:
		raw_idiom = (row.get(idiom_col) or "").strip()
		result = client.count(raw_idiom)
		if result.error:
			n_errors += 1
			if n_errors <= 5:
				logger.warning("Query failed: idiom=%r error=%s", raw_idiom, result.error)
			elif n_errors == 6:
				logger.warning("More query failures encountered; suppressing further per-idiom warnings")

		out_row = dict(row)
		out_row["infinigram_index"] = index
		out_row["infinigram_query"] = raw_idiom
		out_row["infinigram_count"] = result.count
		out_row["infinigram_approx"] = result.approx
		out_row["infinigram_latency"] = result.latency
		out_row["infinigram_token_ids"] = (
			_json_dumps_compact(result.token_ids) if result.token_ids is not None else None
		)
		out_row["infinigram_tokens"] = _json_dumps_compact(result.tokens) if result.tokens is not None else None
		out_row["infinigram_error"] = result.error
		out_row["infinigram_total_tokens"] = total_tokens

		if result.count is not None and total_tokens:
			out_row["infinigram_per_billion_tokens"] = (result.count / total_tokens) * 1e9
		else:
			out_row["infinigram_per_billion_tokens"] = None

		out_rows.append(out_row)

		if tqdm is not None and not disable_tqdm:
			# type: ignore[union-attr]
			iter_rows.set_postfix_str(f"errors={n_errors}")
		elif (i + 1) % 100 == 0:
			logger.info("Progress: %d/%d processed (errors=%d)", i + 1, len(rows), n_errors)

		if sleep_s > 0 and i + 1 < len(rows):
			time.sleep(sleep_s)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames: List[str] = []
	for r in out_rows:
		for k in r.keys():
			if k not in fieldnames:
				fieldnames.append(k)

	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in out_rows:
			writer.writerow(r)

	logger.info("Wrote output: %s (rows=%d, errors=%d)", str(output_path), len(out_rows), n_errors)


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description="Query Infini-gram API counts for idioms and write an enriched CSV."
	)
	p.add_argument(
		"--input",
		required=True,
		type=Path,
		help="Path to idioms CSV/TSV (recommended) or newline-separated text file.",
	)
	p.add_argument(
		"--output",
		required=True,
		type=Path,
		help="Output CSV path.",
	)
	p.add_argument(
		"--column",
		default=None,
		help="Idiom column to query (CSV/TSV only). Defaults to base_form/extracted_idiom/idiom if present.",
	)
	p.add_argument(
		"--index",
		default=INDEX_DEFAULT,
		help=f"Infini-gram index name (default: {INDEX_DEFAULT}).",
	)
	p.add_argument(
		"--api-url",
		default=API_URL_DEFAULT,
		help=f"Infini-gram API URL (default: {API_URL_DEFAULT}).",
	)
	p.add_argument(
		"--timeout-s",
		type=float,
		default=30.0,
		help="HTTP timeout (seconds).",
	)
	p.add_argument(
		"--retries",
		type=int,
		default=6,
		help="Number of retries on transient failures.",
	)
	p.add_argument(
		"--sleep-s",
		type=float,
		default=0.0,
		help="Sleep between requests (seconds).",
	)
	p.add_argument(
		"--log-level",
		default="INFO",
		help="Logging level (DEBUG, INFO, WARNING, ERROR).",
	)
	p.add_argument(
		"--disable-tqdm",
		action="store_true",
		help="Disable tqdm progress bar (useful for non-interactive logs).",
	)
	return p


def main() -> None:
	args = build_arg_parser().parse_args()
	_configure_logging(args.log_level)
	run(
		input_path=args.input,
		output_path=args.output,
		column=args.column,
		index=args.index,
		api_url=args.api_url,
		timeout_s=args.timeout_s,
		retries=args.retries,
		sleep_s=args.sleep_s,
		disable_tqdm=args.disable_tqdm,
	)


if __name__ == "__main__":
	main()
