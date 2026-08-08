from pathlib import Path

path = Path(r"C:\olinecaller\linecaller\dataset\studio_window.py")

text = path.read_text(encoding="utf-8")

text = text.replace(
    "from linecaller.proposals.mock_engine import MockProposalEngine",
    "from linecaller.dataset.proposal_engine_factory import create_dataset_studio_proposal_engine",
)

text = text.replace(
    """        # Mock engine proves UI/API integration.
        # CP-0014C will replace this with the first real AI detector.
        self.controller = AssistedAnnotationController(
            self.session,
            engine=MockProposalEngine(),
        )""",
    """        self.controller = AssistedAnnotationController(
            self.session,
            engine=create_dataset_studio_proposal_engine(),
        )""",
)

path.write_text(text, encoding="utf-8")
print("Dataset Studio proposal engine patched for CP-0014C")
