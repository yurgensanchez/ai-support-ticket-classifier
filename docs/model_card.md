# Model Card

## Model Name

AI Support Ticket Classifier baseline model.

## Objective

Classify short customer support messages into one of six categories:

- sales
- technical_support
- billing
- complaint
- appointment
- other

## Intended Use

This model is intended for portfolio demonstration and local experimentation. It shows how to build a basic supervised text classification pipeline and expose it through an API.

## Not Intended For

- Production customer support routing.
- Automated decisions without human review.
- Sensitive or regulated customer communications.
- Measuring real business performance.

## Data

The first version uses `data/sample_tickets.csv`, a small fictional dataset created for this project. It is not collected from real customers and does not contain private information.

## Preprocessing

- Trim leading and trailing whitespace.
- Convert text to lowercase.
- Collapse repeated whitespace.

## Algorithm

- TF-IDF vectorization with unigrams and bigrams.
- Logistic Regression classifier.

## Metrics

Training writes evaluation output to `models/metrics.json`.

The reported metrics include:

- accuracy
- weighted precision
- weighted recall
- weighted F1
- per-class classification report

## Known Limitations

- The dataset is too small to represent real customer language.
- The model may overfit to simple keywords.
- Similar categories may be confused.
- Confidence values should not be treated as calibrated probabilities.
- The model has not been evaluated on real support conversations.

## Ethical and Practical Considerations

This model should not be used to hide or deprioritize customer complaints. If used in a real setting, it should support human triage rather than replace it.

## Future Improvements

- Add more labeled examples per category.
- Use a real public dataset if appropriate.
- Add confusion matrix output.
- Compare the baseline with embeddings or transformer-based classifiers.
- Add monitoring for low-confidence predictions.
