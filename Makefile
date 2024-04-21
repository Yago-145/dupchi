app::
	cd app && python app_ui_gradio.py

update-pkg::
	poetry build
	cp -f dist/*.whl app/
	rm -rf dist

docker-img::
	cd app && docker build -t dupchi-docker-image .

docker-push::
	docker tag dupchi-docker-image europe-west9-docker.pkg.dev/dupchi/dupchi-docker-repo/dupchi-docker-image
	docker push europe-west9-docker.pkg.dev/dupchi/dupchi-docker-repo/dupchi-docker-image