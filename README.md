# Premium Trading Command Center

Dashboard local em Streamlit para segunda tela, focado em ações de AI/tech, semicondutores, ETFs, Brasil, Bitcoin e macro.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## O que está incluído

- Cards premium por ativo, ordenados por maior alta do dia.
- Auto refresh: 10 segundos durante pregão dos EUA, 60 segundos fora do pregão.
- Header macro com S&P 500, Nasdaq, VIX, Bitcoin, dólar, AUD/USD, minério de ferro e petróleo.
- Radar de oportunidades com breakout, momentum, volume incomum e volatilidade.
- Bitcoin Center com BTC/USD, BTC/BRL, Fear & Greed, dominância, funding rate e níveis técnicos.
- Gráficos Plotly com intraday, 5 dias, 1 mês e 1 ano, média móvel, volume, RSI e MACD.
- AI Market Mood com score bullish/neutro/bearish baseado em momentum, volume, VIX, Nasdaq, Bitcoin e semicondutores.

## Observações

O app usa APIs gratuitas e públicas. Alguns dados como fluxo real de ETFs, short interest intraday e unusual options activity normalmente exigem fornecedores pagos. O dashboard inclui proxies visuais e deixa a arquitetura pronta para integrar APIs premium depois.
