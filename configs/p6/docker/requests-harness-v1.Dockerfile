FROM swebench/sweb.eval.x86_64.psf_1776_requests-5414:latest

USER root
RUN /opt/miniconda3/envs/testbed/bin/python -m pip install --no-cache-dir \
      "pytest>=2.8.0,<=6.2.5" \
      pytest-cov \
      pytest-httpbin==1.0.0 \
      pytest-mock==2.0.0 \
      httpbin==0.7.0 \
      "Flask>=1.0,<2.0" \
      "Jinja2<3" \
      "MarkupSafe<2.1" \
      "Werkzeug<2" \
      "itsdangerous<2" \
      "click<8" \
      trustme \
      wheel \
    && /opt/miniconda3/envs/testbed/bin/python -m pip check
