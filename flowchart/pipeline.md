# 📊 Idea2Paper Pipeline

```mermaid
flowchart TD
    A[Upload Doc(s)] --> B[Text Extraction<br/>(PDF/DOCX/TXT/MD)]
    B --> C[Keyword Extraction]
    C --> D[arXiv Retrieval (API)]
    D --> E[SPECTER Ranking<br/>(Semantic Similarity)]
    E --> F{Background Source?}

    F -->|Full-PDF enabled<br/>+ PDFs found| G[Download & Extract PDFs]
    G --> H[Hierarchical Summarization<br/>(PEGASUS-arXiv)]

    F -->|Fallback / No PDFs| I[Concatenate Titles + Abstracts]
    I --> H

    H --> J[Per-paper One-liners<br/>(PEGASUS)]
    J --> K[Draft Composer]
    K --> L[Markdown Draft + Links]

    %% styling
    style A fill:#E8F0FE,stroke:#3367D6,stroke-width:2px
    style B fill:#E8F0FE,stroke:#3367D6,stroke-width:2px
    style C fill:#FFF7E6,stroke:#FBBC04,stroke-width:2px
    style D fill:#E6F4EA,stroke:#34A853,stroke-width:2px
    style E fill:#E0F7FA,stroke:#00ACC1,stroke-width:2px
    style F fill:#FCE8E6,stroke:#EA4335,stroke-width:2px
    style G fill:#F1F8E9,stroke:#7CB342,stroke-width:2px
    style H fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px
    style I fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px
    style J fill:#F1F8E9,stroke:#7CB342,stroke-width:2px
    style K fill:#F1F8E9,stroke:#7CB342,stroke-width:2px
    style L fill:#FFF3E0,stroke:#FF6F00,stroke-width:2px