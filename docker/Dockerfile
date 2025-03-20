FROM debian:12

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        python3-pyqt5.qtsvg

ENV PATH="/root/.local/bin:$PATH"

RUN wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3 /tmp/get-pip.py --user --break-system-packages

COPY requirements.txt /tmp

RUN pip3 install --user --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# use the default bashrc provided by Ubuntu & change the color to yellow
RUN cp /etc/skel/.bashrc /etc/bash.bashrc && \
    sed -i 's/#force_color_prompt/force_color_prompt/g' /etc/bash.bashrc && \
    sed -i 's/32m/33m/g' /etc/bash.bashrc && \
    echo "\nalias pylab='ipython --pylab'" >> /etc/bash.bashrc

RUN find /root -type d -exec chmod 755 {} +

ENV PYTHONPATH=/root/.local/lib/python3.11/site-packages

RUN rm -rf /tmp/* /var/lib/apt/lists/*
