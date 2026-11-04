from collections.abc import Sequence

CORPUS_PREPARERS = {}


def preparers_for(corpus: str) -> list[type]:
    """Return the registered preparer classes for ``corpus`` as a list."""
    value = CORPUS_PREPARERS[corpus]
    if isinstance(value, type):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise TypeError(f"Registry value for {corpus!r} must be a class or sequence of classes, got {type(value).__name__}")


from speechlmm.data.processing.ami.prepare import (AmiCorpusPreparer, AmiAVSpeakDCorpusPreparer, AmiAChapCorpusPreparer)
CORPUS_PREPARERS.update({"ami": [AmiCorpusPreparer, AmiAChapCorpusPreparer, AmiAVSpeakDCorpusPreparer]})

from speechlmm.data.processing.common_voice.prepare import (CommonVoiceCorpusPreparer)
CORPUS_PREPARERS.update({"common_voice": [CommonVoiceCorpusPreparer]})

from speechlmm.data.processing.dgt_tm.prepare import (DGTTMCorpusPreparer)
CORPUS_PREPARERS.update({"dgt_tm": [DGTTMCorpusPreparer]})

from speechlmm.data.processing.elitr.prepare import (ELITRCorpusPreparer)
CORPUS_PREPARERS.update({"elitr": [ELITRCorpusPreparer]})

from speechlmm.data.processing.europarl_st.prepare import (EuroparlSTCorpusPreparer)
CORPUS_PREPARERS.update({"europarl_st": [EuroparlSTCorpusPreparer]})

from speechlmm.data.processing.fleurs.prepare import (FleursCorpusPreparer)
CORPUS_PREPARERS.update({"fleurs": [FleursCorpusPreparer]})

from speechlmm.data.processing.icsi.prepare import ( ICSIUnsegmentedCorpusPreparer, ICSISegmentedCorpusPreparer)
CORPUS_PREPARERS.update({"icsi": [ICSISegmentedCorpusPreparer, ICSIUnsegmentedCorpusPreparer]})

from speechlmm.data.processing.iwslt26.prepare import (IWSLT26CorpusPreparer)
CORPUS_PREPARERS.update({"iwslt26": [IWSLT26CorpusPreparer]})

from speechlmm.data.processing.librisqa.prepare import (LIBRISQACorpusPreparer)
CORPUS_PREPARERS.update({"librisqa": [LIBRISQACorpusPreparer]})

from speechlmm.data.processing.lipcrops.prepare import (LipCropsCorpusPreparer)
CORPUS_PREPARERS.update({"lipcrops": [LipCropsCorpusPreparer]})

from speechlmm.data.processing.slurp.prepare import (SlurpCorpusPreparer)
CORPUS_PREPARERS.update({"slurp": [SlurpCorpusPreparer]})

from speechlmm.data.processing.topical_chat.prepare import (TopicalChatCorpusPreparer)
CORPUS_PREPARERS.update({"topical_chat": [TopicalChatCorpusPreparer]})

from speechlmm.data.processing.speechmassive.prepare import (SpeechmassiveCorpusPreparer)
CORPUS_PREPARERS.update({"speechmassive": [SpeechmassiveCorpusPreparer]})

from speechlmm.data.processing.spokensquad.prepare import (SpokenSquadCorpusPreparer)
CORPUS_PREPARERS.update({"spokensquad": [SpokenSquadCorpusPreparer]})

from speechlmm.data.processing.voxpopuli_eurospeech.prepare import (VPESCorpusPreparer)
CORPUS_PREPARERS.update({"voxpopuli_eurospeech": [VPESCorpusPreparer]})

from speechlmm.data.processing.ytseg_cc_by.prepare import (YTSegCCByCorpusPreparer)
CORPUS_PREPARERS.update({"ytseg_cc_by": [YTSegCCByCorpusPreparer]})

from speechlmm.data.processing.ytseg.prepare import (YTSegCorpusPreparer)
CORPUS_PREPARERS.update({"ytseg": [YTSegCorpusPreparer]})
