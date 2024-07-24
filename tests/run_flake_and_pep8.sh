cd ..
flake8 --config=tests/.flake8 --format=pylint --tee --output-file=tests/report-flake8.txt **/*.py > flake8_results.txt
flake8 --config=tests/.flake8 --format=pylint --tee --output-file=tests/report-flake8.txt **/**/*.py >> flake8_results.txt
pylint **/*.py > pylint_results.txt
pylint **/**/*.py >> pylint_results.txt