# 📚 Índice Central - Embeddings Particionados

**Última atualização:** 18 de maio de 2026  
**Versão:** 1.0  
**Status:** ✅ Pronto para produção

---

## 🚀 Início Rápido

**TL;DR:** Mude o nome do script de `run_embeddings_podman.sh` para `run_embeddings_partitioned_podman.sh`

```bash
# ANTES
./run_linktransformer/run_embeddings_podman.sh --mode base --left-on ...

# DEPOIS
./run_linktransformer/run_embeddings_partitioned_podman.sh --mode base --left-on ...
```

Pronto! Agora embeddings serão processados em partições de 100k linhas.

---

## 📚 Documentação por Nível

### ⚡ Para Pressa (2 minutos)
→ [QUICK_REFERENCE.md](run_linktransformer/QUICK_REFERENCE.md)
- Comandos mais comuns
- Troubleshooting rápido
- Referência de opções

### 🎯 Para Começar (10 minutos)
→ [README_ALTERACOES_EMBEDDINGS.md](run_linktransformer/README_ALTERACOES_EMBEDDINGS.md)
- O que foi feito
- Benefícios principais
- Estrutura de saída
- Exemplos básicos

### 📖 Para Entender (30 minutos)
→ [EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md](run_linktransformer/EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md)
- 6 exemplos práticos diferentes
- Casos de uso específicos
- Monitoramento em tempo real
- Validação de resultado

### 🔧 Para Integrar (45 minutos)
→ [GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md](run_linktransformer/GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md)
- Compatibilidade com código antigo
- Integração em pipeline
- Uso de partições em produção
- Recuperação de falhas
- Otimizações avançadas

### 📘 Para Aprender (1-2 horas)
→ [EMBEDDINGS_PARTICIONADOS.md](EMBEDDINGS_PARTICIONADOS.md)
- Documentação técnica completa
- Todas as opções em detalhes
- Estrutura interna
- Formato de manifest
- FAQ completo

---

## 📁 Arquivos Criados

### 🐍 Scripts Python

| Arquivo | Descrição | Tamanho |
|---------|-----------|--------|
| [run_embeddings_partitioned.py](run_linktransformer/run_embeddings_partitioned.py) | ⭐ Script principal | ~600 linhas |
| [merge_partitions.py](run_linktransformer/merge_partitions.py) | Mesclar partições manualmente | ~150 linhas |
| [test_embeddings_partitioned.sh](run_linktransformer/test_embeddings_partitioned.sh) | Script de teste rápido | ~100 linhas |

### 🔧 Scripts Shell

| Arquivo | Descrição |
|---------|-----------|
| [run_embeddings_partitioned_podman.sh](run_linktransformer/run_embeddings_partitioned_podman.sh) | Wrapper Podman/Docker |

### 📄 Documentação

| Arquivo | Público | Tempo Leitura | Propósito |
|---------|---------|--------------|----------|
| [QUICK_REFERENCE.md](run_linktransformer/QUICK_REFERENCE.md) | ⚡ Pressa | 2 min | Referência rápida |
| [README_ALTERACOES_EMBEDDINGS.md](run_linktransformer/README_ALTERACOES_EMBEDDINGS.md) | 🎯 Iniciante | 10 min | Overview das mudanças |
| [EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md](run_linktransformer/EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md) | 📖 Aprendiz | 30 min | Exemplos práticos |
| [GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md](run_linktransformer/GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md) | 🔧 Integrador | 45 min | Como integrar |
| [EMBEDDINGS_PARTICIONADOS.md](EMBEDDINGS_PARTICIONADOS.md) | 📘 Completo | 1-2h | Documentação técnica |

---

## 🎯 Guia de Seleção

### Você quer...

**... executar o comando rapidinho?**
- → Ler: [QUICK_REFERENCE.md](run_linktransformer/QUICK_REFERENCE.md) (2 min)
- → Comando: `./run_linktransformer/run_embeddings_partitioned_podman.sh --mode base --left-on ...`

**... entender o que mudou?**
- → Ler: [README_ALTERACOES_EMBEDDINGS.md](run_linktransformer/README_ALTERACOES_EMBEDDINGS.md) (10 min)
- → Ver: Seção "Comparação Antes/Depois"

**... usar partições em casos específicos?**
- → Ler: [EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md](run_linktransformer/EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md) (30 min)
- → Escolher exemplo mais próximo do seu caso
- → Copiar e adaptar comando

**... integrar no seu pipeline?**
- → Ler: [GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md](run_linktransformer/GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md) (45 min)
- → Seção "Integração com Código Existente"
- → Copiar snippets Python/Bash

**... entender tudo em detalhes?**
- → Ler: [EMBEDDINGS_PARTICIONADOS.md](EMBEDDINGS_PARTICIONADOS.md) (1-2h)
- → Todas as opções, estruturas, etc.

---

## 🧪 Testar

Executar teste rápido (1-2 minutos):

```bash
./run_linktransformer/test_embeddings_partitioned.sh
```

Isso vai:
- ✓ Criar CSV de teste (350 linhas)
- ✓ Processar em 4 partições de 100 linhas
- ✓ Gerar embeddings (modelo rápido)
- ✓ Mesclar partições
- ✓ Validar resultado

---

## 📊 Estrutura de Saída

```
pasta_aux/
├── embeddings_partitions/
│   ├── modelo_1/
│   │   ├── partition_000000_base.npy    ← 100k linhas
│   │   ├── partition_000001_base.npy    ← 100k linhas
│   │   └── partition_000002_base.npy    ← resto
│   └── modelo_2/
│       └── ... (similar)
├── embeddings_base_modelo_1.npy         ← Mesclado
├── embeddings_base_modelo_2.npy         ← Mesclado
└── embeddings_manifest.json             ← Metadados
```

---

## 🔑 Principais Conceitos

### Partições
- Arquivo CSV dividido em chunks de 100k linhas
- Cada partição processada independentemente
- Salvo em arquivo `.npy` separado

### Pastas por Modelo
```
embeddings_partitions/
├── sentence-transformers_all-MiniLM-L6-v2/
├── intfloat_multilingual-e5-large/
└── ... (outros modelos)
```

### Mescla Automática
- Com `--merge-partitions`: concatena partições em arquivo único
- Compatível com código antigo que espera `embeddings_base_*.npy`

### Manifest JSON
```json
{
  "model": "...",
  "partition_size": 100000,
  "num_partitions": 3,
  "elapsed_seconds": 1234.56,
  "memory_delta_mb": 1536.0
}
```

---

## ⚙️ Opções Principais

```bash
--partition-size SIZE       # Linhas por partição (padrão: 100000)
--merge-partitions          # Mesclar em arquivo único
--batch-size SIZE           # Batch (padrão: 128)
--mode {base|query|both}    # Qual lado (padrão: both)
--model MODELO              # Modelo específico (repetir para vários)
--manifest-path PATH        # Onde salvar manifest
```

Exemplos:

```bash
# Partições menores (baixa memória)
--partition-size 50000 --batch-size 32

# Sem mesclar (economiza disco)
--no-merge-partitions

# Modelo específico
--model sentence-transformers/all-MiniLM-L6-v2
```

---

## 🚀 Próximos Passos

### 1️⃣ Teste Rápido (1-2 min)
```bash
./run_linktransformer/test_embeddings_partitioned.sh
```

### 2️⃣ Seu Primeiro Comando
```bash
HOST_DATA_DIR="$(pwd)/pasta_aux" ./run_linktransformer/run_embeddings_partitioned_podman.sh \
  --mode base \
  --left-on uf municipio logradouro numero complemento localidade setor_censitario
```

### 3️⃣ Monitorar
```bash
tail -f embeddings_base_pasta_aux.log
```

### 4️⃣ Validar
```python
import numpy as np
embeddings = np.load('pasta_aux/embeddings_base_*.npy')
print(f"Shape: {embeddings.shape}")  # (N, 384)
```

---

## 🐛 Troubleshooting Rápido

| Problema | Solução |
|----------|---------|
| Out of Memory | `--batch-size 32 --partition-size 50000` |
| Disco Cheio | `--no-merge-partitions` |
| Muito Lento | `--batch-size 256` |
| Partições Incompletas | Reexecutar (sobrescreve) |
| Módulo não encontrado | Verificar `sys.path` no script |

Mais detalhes: [QUICK_REFERENCE.md](run_linktransformer/QUICK_REFERENCE.md#troubleshooting-rápido)

---

## 📋 Checklist de Implementação

- ✅ `run_embeddings_partitioned.py` criado e testado
- ✅ `run_embeddings_partitioned_podman.sh` criado
- ✅ `merge_partitions.py` utilitário criado
- ✅ `test_embeddings_partitioned.sh` para teste
- ✅ Containerfile.embeddings atualizado
- ✅ Documentação completa (5 arquivos)
- ✅ Exemplos práticos
- ✅ Guia de integração
- ✅ Validação de sintaxe Python
- ✅ Testes de import

---

## 📞 Suporte

### Para cada situação:

**"Como executo?"**
→ [QUICK_REFERENCE.md](run_linktransformer/QUICK_REFERENCE.md)

**"O que mudou?"**
→ [README_ALTERACOES_EMBEDDINGS.md](run_linktransformer/README_ALTERACOES_EMBEDDINGS.md)

**"Tenho um caso específico"**
→ [EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md](run_linktransformer/EXEMPLOS_USO_EMBEDDINGS_PARTICIONADOS.md)

**"Preciso integrar no meu código"**
→ [GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md](run_linktransformer/GUIA_INTEGRACAO_EMBEDDINGS_PARTICIONADOS.md)

**"Preciso entender tudo"**
→ [EMBEDDINGS_PARTICIONADOS.md](EMBEDDINGS_PARTICIONADOS.md)

---

## 📈 Roadmap (Futuro)

Possíveis melhorias:

- [ ] Suporte para modo "query" com particionamento
- [ ] Processamento paralelo automático de modelos
- [ ] Compressão de partições (.npy.gz)
- [ ] Cache de embeddings processados
- [ ] Interface web para monitoramento
- [ ] Suporte para mais formatos (Parquet, Arrow, etc.)

---

**Versão:** 1.0  
**Data:** 18 de maio de 2026  
**Autor:** GitHub Copilot

---

**🎉 Tudo pronto! Comece por [QUICK_REFERENCE.md](run_linktransformer/QUICK_REFERENCE.md)**
