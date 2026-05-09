.PHONY: test api frontend space

test:
	./scripts/pytest.sh -q

api:
	./scripts/run_api.sh

frontend:
	./scripts/run_frontend.sh

space:
	streamlit run frontend/app_space.py
