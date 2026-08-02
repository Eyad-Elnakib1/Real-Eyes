"""
utils/trust.py
--------------
Unified source-trust scoring used across all three layers:

    crawler/frontier.py   – trusted domains crawled first
    search/retriever.py   – trust blended into final ranking
    test_live_web3.py     – credibility weighting for fact-check verdicts

Scoring precedence (highest wins):
    1. explicit domain match   (hand-curated _DOMAIN_TRUST)
    2. TLD / pattern match      (.gov, .edu, .ac.*, .int, .mil, .org)
    3. spam / low-authority penalty (free hosts, user-generated platforms)
    4. default 1.0 (unknown / neutral)

This module is dependency-free (stdlib only) so it can be imported both
inside the semantic_crawler package and from the root-level scripts.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Optional, Union

# Final trust multipliers are clamped to this range.
TRUST_MIN = 0.2
TRUST_MAX = 3.0
TRUST_NEUTRAL = 1.0

# WP:RSP is the trust *baseline* for general media: any source Wikipedia's
# community classifies as generally-reliable gets RSP_TRUSTED_WEIGHT, which
# already beats an unknown source (1.0). Only the objective top tiers in
# _DOMAIN_TRUST (fact-checkers, wire services, official authorities) sit above
# it. Unreliable/deprecated/blacklisted sources are floored.
RSP_TRUSTED_WEIGHT = 2.0
RSP_BLOCKED_WEIGHT = TRUST_MIN

# ── 1. Objective high-authority overrides (above the WP:RSP baseline) ───────────
# Only sources whose weight is NOT a judgment call belong here: dedicated
# fact-checkers, wire services, official government/health/science bodies,
# peer-reviewed journals, and core reference works — plus social/UGC penalties.
# General news, magazines, tech & entertainment outlets are deliberately NOT
# hand-weighted here; their trust comes from the community-vetted WP:RSP list
# below at a flat baseline (RSP_TRUSTED_WEIGHT), so we never invent a number
# for a source Wikipedia has already classified.
# https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources
# Emptied by request: trust is driven entirely by the WP:RSP lists below
# (generally-reliable → RSP_TRUSTED_WEIGHT, unreliable → floored), with the
# .gov/.int/.edu TLD rules as a fallback for official sources WP:RSP omits.
# To re-introduce a hand-weighted top tier (e.g. fact-checkers at 3.0), add
# entries here — they take precedence over the WP:RSP baseline.
_DOMAIN_TRUST: dict[str, float] = {}

# ── 1a. WP:RSP bulk lists (hand-maintained from the Perennial Sources page) ─────
# The community-vetted trust baseline. Domains Wikipedia lists as "generally
# reliable" all get a flat RSP_TRUSTED_WEIGHT (we don't second-guess Wikipedia
# with invented per-domain numbers); deprecated / blacklisted / generally-
# unreliable domains are floored. The small objective top tiers in _DOMAIN_TRUST
# (checked first) are the only sources allowed to outrank this baseline.
# Source: https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources
# To update: edit these sets by hand (add/remove a domain as WP:RSP changes).
RSP_TRUSTED: frozenset[str] = frozenset({
    "abcnews.com", "abcnews.go.com", "adl.org", "afp.com", "aljazeera.com",
    "aljazeera.net", "amnesty.org", "ap.org", "apnews.com", "arstechnica.com",
    "arstechnica.co.uk", "atlasobscura.com", "avclub.com", "axios.com",
    "ballotpedia.org", "bbc.co.uk", "bbc.com", "bellingcat.com", "bloomberg.com",
    "businessweek.com", "buzzfeednews.com", "cbsnews.com", "channelnewsasia.com",
    "checkyourfact.com", "climatefeedback.org", "cnet.com", "cnn.com",
    "codastory.com", "commonsensemedia.org", "csmonitor.com", "deadline.com",
    "denofgeek.com", "deseret.com", "deseretnews.com", "digitalspy.com",
    "digitaltrends.com", "dw.com", "economist.com", "engadget.com",
    "eurogamer.net", "ew.com", "forbes.com", "freebeacon.com", "ft.com",
    "gamasutra.com", "gamedeveloper.com", "gameinformer.com", "gamespot.com",
    "gizmodo.com", "glaad.org", "gq.com", "gq-magazine.co.uk", "grubstreet.com",
    "guardian.co.uk", "haaretz.com", "haaretz.co.il", "hollywoodreporter.com",
    "hopenothate.org.uk", "huffpost.com", "huffingtonpost.com",
    "huffingtonpost.co.uk", "huffingtonpost.ca", "ign.com", "independent.co.uk",
    "indianexpress.com", "insider.com", "ipsnews.net", "iranicaonline.org",
    "jamanetwork.com", "journalism.org", "jpost.com", "kirkusreviews.com",
    "kommersant.ru", "latimes.com", "lwn.net", "meduza.io", "metacritic.com",
    "metro.co.uk", "mg.co.za", "mondediplo.com", "monde-diplomatique.fr",
    "motherjones.com", "msnbc.com", "nationalgeographic.com", "nationalpost.com",
    "nbcnews.com", "newrepublic.com", "news.sky.com", "newslaundry.com",
    "newsnationnow.com", "newsweek.com", "newyorker.com", "nme.com", "npr.org",
    "nydailynews.com", "nymag.com", "nytimes.com", "nzherald.co.nz", "oko.press",
    "pbs.org", "people.com", "pewresearch.org", "pinknews.co.uk", "politico.com",
    "politifact.com", "polygon.com", "propublica.org", "rappler.com", "reason.com",
    "reuters.com", "rfa.org", "rollingstone.com", "rottentomatoes.com",
    "scientificamerican.com", "scmp.com", "scotusblog.com", "si.com",
    "sixthtone.com", "skepticalinquirer.org", "smh.com.au", "snopes.com",
    "space.com", "spiegel.de", "splcenter.org", "straitstimes.com",
    "telegraph.co.uk", "theage.com.au", "theatlantic.com", "theaustralian.com.au",
    "theconversation.com", "thecut.com", "thediplomat.com", "theglobeandmail.com",
    "theguardian.com", "theguardian.co.uk", "thehill.com", "thehindu.com",
    "theintercept.com", "themarysue.com", "thenation.com", "theregister.com",
    "theregister.co.uk", "thetimes.co.uk", "thetimes.com", "theverge.com",
    "thewire.in", "thewrap.com", "time.com", "timesofisrael.com", "torrentfreak.com",
    "tvguide.com", "usatoday.com", "usnews.com", "vanityfair.com", "variety.com",
    "villagevoice.com", "voanews.com", "vogue.com", "vox.com", "vulture.com",
    "washingtonpost.com", "wired.com", "wired.co.uk", "wsj.com", "wyborcza.pl",
    "zdnet.com",
})

# Deprecated / blacklisted / generally-unreliable on WP:RSP. (Wikipedia itself,
# Wikidata, etc. are deliberately NOT here — the pipeline uses Wikipedia as an
# evidence source, even though WP:RSP discourages *citing* it.)
RSP_BLOCKED: frozenset[str] = frozenset({
    "112.international", "112.ua", "aa.com.tr", "actualidad-rt.com",
    "adfontesmedia.com", "almanar.com.lb", "almayadeen.net", "almayadeen.tv",
    "alternet.org", "amazon.com", "amazon.co.uk", "amazon.ca", "amazon.de",
    "ancestry.com", "anna-news.info", "antiwar.com", "armyrecognition.com",
    "arxiv.org", "askubuntu.com", "baike.baidu.com", "banned.video",
    "benzinga.com", "bestgore.com", "biggovernment.com", "bild.de", "biorxiv.org",
    "bitterwinter.org", "blogspot.com", "breitbart.com", "broadwayworld.com",
    "californiaglobe.com", "celebritynetworth.com", "cesnur.org", "cgtn.com",
    "change.org", "cnsnews.com", "coindesk.com", "company-histories.com",
    "conservativereview.com", "consortiumnews.com", "counterpunch.org",
    "cracked.com", "crunchbase.com", "dailycaller.com", "dailykos.com",
    "dailymail.co.uk", "dailymail.com", "dailysabah.com", "dailystar.co.uk",
    "dailywire.com", "deviantart.com", "dexerto.com", "discogs.com",
    "distractify.com", "eadaily.com", "electronicintifada.net",
    "epochtimes.com", "ethnicelebs.com", "eurasiantimes.com", "examiner.com",
    "express.co.uk", "facebook.com", "familysearch.org", "famousbirthdays.com",
    "fandom.com", "faroutmagazine.co.uk", "filmaffinity.com", "findagrave.com",
    "findmypast.com", "flickr.com", "foxnews.com", "frontpagemag.com",
    "fundinguniverse.com", "gamepedia.com", "gawker.com", "gbnews.com",
    "geni.com", "globalresearch.ca", "globalsecurity.org", "globaltimes.cn",
    "gofundme.com", "goodreads.com", "healthline.com", "heatst.com",
    "heritage.org", "hispantv.com", "history.com", "huanqiu.com", "ibtimes.com",
    "ibtimes.co.uk", "imc-africa.mayfirst.org", "imdb.com", "inaturalist.org",
    "indiegogo.com", "indymedia.org", "infowars.com", "infowars.net",
    "inquisitr.com", "instagram.com", "investopedia.com", "jewishvirtuallibrary.org",
    "jihadwatch.org", "joshuaproject.net", "journal-neo.org",
    "journalofscientificexploration.org", "kickstarter.com", "knowyourmeme.com",
    "ladbible.com", "last.fm", "lenta.ru", "lifesitenews.com", "linkedin.com",
    "liveleak.com", "livejournal.com", "lulu.com", "marquiswhoswho.com",
    "mashable.com", "mathoverflow.net", "meaww.com", "mediabiasfactcheck.com",
    "medium.com", "medrxiv.org", "metal-archives.com", "mintpressnews.com",
    "mondialisation.ca", "mrc.org", "mrctv.org", "mylife.com",
    "nationalenquirer.com", "nationalfile.com", "naturalnews.com",
    "navyrecognition.com", "news-front.info", "newsblaze.com", "newsbreak.com",
    "newsbusters.org", "newsmax.com", "newsmaxtv.com", "newsoftheworld.co.uk",
    "newstarget.com", "newswars.com", "ngo-monitor.org", "nndb.com",
    "nspirement.com", "ntd.com", "ntdtv.com", "nypost.com", "oann.com",
    "occupydemocrats.com", "okeefemediagroup.com", "opindia.com", "order-order.com",
    "ourcampaigns.com", "pagesix.com", "panampost.com", "patheos.com",
    "pinkvilla.com", "planespotters.net", "pressreader.com", "presstv.com",
    "presstv.ir", "prisonplanet.com", "prnewswire.com", "projectveritas.com",
    "quadrant.org.au", "quillette.com", "quora.com", "rateyourmusic.com",
    "rawstory.com", "reddit.com", "redfish.media", "redstate.com",
    "republicworld.com", "rogueimc.org", "royalcentral.co.uk", "rt.com", "rt.rs",
    "ruptly.tv", "russiatoday.com", "sarabic.ae", "sciencedirect.com",
    "scientificexploration.org", "scribd.com", "secretchina.com", "serverfault.com",
    "simpleflying.com", "skwawkbox.org", "sourcewatch.org", "southfront.org",
    "sportskeeda.com", "sputnik.by", "sputnikglobe.com", "sputniknews.com",
    "sputniknews.ru", "stackexchange.com", "stackoverflow.com", "starsunfolded.com",
    "statista.com", "superuser.com", "swarajyamag.com", "swentr.site", "takimag.com",
    "tasnimnews.com", "tass.com", "tass.ru", "telesurenglish.net", "the-sun.com",
    "theblaze.com", "thecanary.co", "thecradle.co", "thedebrief.org",
    "theepochtimes.com", "thefederalist.com", "thegatewaypundit.com",
    "thegrayzone.com", "thenewamerican.com", "theonion.com", "thepointsguy.com",
    "thepostmillennial.com", "thesun.co.uk", "thesun.ie", "thetruthaboutguns.com",
    "thisismoney.co.uk", "tiktok.com", "tvtropes.org", "twitter.com", "unz.com",
    "urbandictionary.com", "vdare.com", "venezuelanalysis.com", "veteranstoday.com",
    "vgchartz.com", "victimsofcommunism.org", "visiontimes.com", "voltairenet.org",
    "vtforeignpolicy.com", "washingtonpress.com", "watchmojo.com",
    "wegotthiscovered.com", "wenweipo.com", "westernjournal.com", "whatculture.com",
    "whosampled.com", "wikia.com", "wikidata.org", "wikileaks.org", "wikinews.org",
    "wnd.com", "wordpress.com", "worldnetdaily.com", "worldometers.info", "x.com",
    "youtube.com", "zeenews.india.com", "zerohedge.com", "zoominfo.com",
})

# ── 1c. Arabic trusted sources (beyond WP:RSP) ──────────────────────────────────
# WP:RSP is English-centric, so Arabic regional claims have no trusted coverage
# from RSP_TRUSTED alone. These Arabic fact-checkers + reputable Arabic-language
# outlets get the trusted weight and are included in the trusted-first search for
# the Arabic pipeline. Curated by hand — trim/extend as needed.
TRUSTED_AR: frozenset[str] = frozenset({
    # Arabic fact-checkers
    "misbar.com", "fatabyyano.net",
    # International Arabic-language news
    "alarabiya.net", "skynewsarabia.com", "france24.com",
    "alhurra.com", "independentarabia.com",
    # Major regional dailies (mainstream coverage of local events)
    "almasryalyoum.com", "youm7.com", "asharqalawsat.com", "alquds.co.uk",
})

# ── 2. TLD / pattern trust (fallback when domain not explicitly listed) ─────────
# Checked as substrings/suffixes of the registered domain.
_TLD_TRUST: list[tuple[str, float]] = [
    (".gov", 2.5),     # government
    (".mil", 2.4),     # military
    (".int", 2.4),     # international treaty orgs
    (".edu", 2.3),     # US educational
    (".ac.", 2.3),     # academic (.ac.uk, .ac.jp, …)
    (".edu.", 2.3),    # educational (.edu.au, …)
    (".gov.", 2.5),    # government (.gov.uk, …)
    (".org", 1.3),     # nonprofits / research (mild boost)
]

# ── 3. Spam / low-authority signals (multiplicative penalties) ──────────────────
_SPAM_TLDS = (".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click")
_SPAM_TLD_PENALTY = 0.3

_UGC_HOSTS = ("blogspot.", "wordpress.com", "medium.com", "substack.com",
              "tumblr.com", "wixsite.com", "weebly.com", "blogger.com")
_UGC_PENALTY = 0.6

# ── Clickbait detection ─────────────────────────────────────────────────────────
_CLICKBAIT_PHRASES = (
    "you won't believe", "you wont believe", "shocking", "doctors hate",
    "this one trick", "what happens next", "will blow your mind",
    "gone wrong", "number 7 will", "the truth about", "they don't want you",
    "miracle", "secret", "exposed", "jaw-dropping",
)


def extract_registered_domain(url: str) -> str:
    """Return the lowercased host without a leading 'www.'.

    Not a full public-suffix parse — good enough for trust lookups.
    """
    host = url
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    host = host.split(":", 1)[0]  # strip port
    host = host.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def trust_score(url: str) -> float:
    """Trust multiplier in [TRUST_MIN, TRUST_MAX].

    Precedence: explicit domain > TLD pattern > default, then spam penalties
    are applied multiplicatively (unless an explicit low score already set it).
    """
    domain = extract_registered_domain(url)
    if not domain:
        return TRUST_NEUTRAL

    # 1. Explicit domain match (exact or subdomain of a listed domain)
    for key, w in _DOMAIN_TRUST.items():
        if domain == key or domain.endswith("." + key):
            return _clamp(w)

    # 1b. WP:RSP lists (generated). Blocked first so an unreliable source can't
    # be rescued by, say, a generic .org TLD boost. Subdomain-aware but cheap:
    # test the host and each parent domain (edition.cnn.com → cnn.com).
    if _in_rsp(domain, RSP_BLOCKED):
        return _clamp(RSP_BLOCKED_WEIGHT)
    if _in_rsp(domain, RSP_TRUSTED) or _in_rsp(domain, TRUSTED_AR):
        return _clamp(RSP_TRUSTED_WEIGHT)

    # 2. TLD / pattern match
    base = TRUST_NEUTRAL
    for pattern, w in _TLD_TRUST:
        if pattern in domain or domain.endswith(pattern.rstrip(".")):
            base = w
            break

    # 3. Spam / low-authority penalties
    if domain.endswith(_SPAM_TLDS):
        base *= _SPAM_TLD_PENALTY
    if any(h in domain for h in _UGC_HOSTS):
        base *= _UGC_PENALTY

    return _clamp(base)


def normalized_trust(url: str) -> float:
    """Trust scaled to [0, 1] (for blending into ranking scores)."""
    return _clamp(trust_score(url)) / TRUST_MAX


def crawl_trust_boost(url: str) -> float:
    """Multiplier for crawl-frontier priority, centred so a 1.5-trust domain
    is neutral (≈1.0). Capped at 2.0 so trust can't completely dominate
    relevance."""
    return min(trust_score(url) / 1.5, 2.0)


def clickbait_penalty(title: str) -> float:
    """Return a multiplier in (0, 1]. 1.0 = clean, lower = more clickbaity."""
    if not title:
        return 1.0
    t = title.lower()
    penalty = 1.0
    if any(p in t for p in _CLICKBAIT_PHRASES):
        penalty *= 0.5
    letters = [c for c in title if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
        penalty *= 0.7  # SHOUTING TITLE
    if title.count("!") >= 2 or title.count("?") >= 2:
        penalty *= 0.8
    return penalty


def freshness_score(published: Union[str, float, datetime, None],
                    half_life_days: float = 365.0) -> float:
    """Exponential-decay freshness in [0, 1].

    *published* may be an ISO date string, a unix timestamp, a datetime, or
    None. Returns 0.5 (neutral) when the date is unknown, so undated pages
    are neither rewarded nor heavily punished.
    """
    ts = _to_timestamp(published)
    if ts is None:
        return 0.5
    age_days = max((time.time() - ts) / 86400.0, 0.0)
    return math.exp(-age_days / half_life_days)


# ── helpers ─────────────────────────────────────────────────────────────────────

def _clamp(x: float) -> float:
    return max(TRUST_MIN, min(TRUST_MAX, x))


def _in_rsp(domain: str, table: frozenset[str]) -> bool:
    """True if `domain` or any parent domain is in a WP:RSP set.

    O(number of labels) membership tests against an O(1) frozenset, so
    edition.cnn.com matches a listed cnn.com without scanning the whole set.
    """
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        if ".".join(parts[i:]) in table:
            return True
    return False


def _to_timestamp(published: Union[str, float, datetime, None]) -> Optional[float]:
    if published is None or published == "":
        return None
    if isinstance(published, (int, float)):
        return float(published)
    if isinstance(published, datetime):
        return published.timestamp()
    # string — try ISO first, then a few common formats
    s = str(published).strip()
    iso = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        pass
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3]),
                            tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None
    return None