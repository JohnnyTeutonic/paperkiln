# Audit sheet

Edit `audit.jsonl`, not this file. For each item reviewed set `audited: true`, `audit_reward` (0 or 1),
`audit_rationale` (why) and `annotator`. Then run `audit_sample.py merge`.

Strata (evidence kind, reward bucket, accepted) -> items: ('exact_match', '0', False) -> 7; ('exact_match', '0', True) -> 1; ('exact_match', '1', False) -> 2; ('exact_match', '1', True) -> 2

## a0000  `exact_match`  machine reward 0.0

- prompt: ben had 3 rocks . then he gets 11 more . how many rocks does he have ?
- completion: the answer is 16 .
- machine rationale: final answer 16 does not match reference 14
- evidence: `tinyarith_r20_v1/dev/add-3-11`

## a0001  `exact_match`  machine reward 0.0

- prompt: anna has 6 cars . she finds 1 cars . how many cars does she have now ?
- completion: the answer is 9 .
- machine rationale: final answer 9 does not match reference 7
- evidence: `tinyarith_r20_v1/dev/add-6-1`

## a0002  `exact_match`  machine reward 0.0

- prompt: sam has 6 apples . he picks up 11 apples . how many apples does he have now ?
- completion: the answer is 19 .
- machine rationale: final answer 19 does not match reference 17
- evidence: `tinyarith_r20_v1/dev/add-6-11`

## a0003  `exact_match`  machine reward 0.0

- prompt: sue has 11 toys . she finds 6 toys . how many toys does she have now ?
- completion: the answer is 19 .
- machine rationale: final answer 19 does not match reference 17
- evidence: `tinyarith_r20_v1/dev/add-11-6`

## a0004  `exact_match`  machine reward 0.0

- prompt: sam has 13 flowers . he drops 5 flowers . how many flowers does he have now ?
- completion: the answer is 9 .
- machine rationale: final answer 9 does not match reference 8
- evidence: `tinyarith_r20_v1/dev/sub-13-5`

## a0005  `exact_match`  machine reward 0.0

- prompt: one day mia has 20 cookies . she loses 2 of them . how many cookies are left ?
- completion: the answer is 2 .
- machine rationale: final answer 2 does not match reference 18
- evidence: `tinyarith_r20_v1/dev/sub-20-2`

## a0006  `exact_match`  machine reward 0.0

- prompt: one day sue has 20 stickers . she drops 19 of them . how many stickers are left ?
- completion: the answer is 0 .
- machine rationale: final answer 0 does not match reference 1
- evidence: `tinyarith_r20_v1/dev/sub-20-19`

## a0007  `exact_match`  machine reward 0.0

- prompt: one day max has 17 cookies . he drops 9 of them . how many cookies are left ?
- completion: the answer is 9 .
- machine rationale: final answer 9 does not match reference 8
- evidence: `tinyarith_r20_v1/dev/sub-17-9`

## a0008  `exact_match`  machine reward 1.0

- prompt: anna had 16 toys . then she finds 4 more . how many toys does she have ?
- completion: the answer is 20 .
- machine rationale: final answer 20 matches reference 20
- evidence: `tinyarith_r20_v1/dev/add-16-4`

## a0009  `exact_match`  machine reward 1.0

- prompt: one day ben has 13 flowers . he loses 11 of them . how many flowers are left ?
- completion: the answer is 2 .
- machine rationale: final answer 2 matches reference 2
- evidence: `tinyarith_r20_v1/dev/sub-13-11`

## a0010  `exact_match`  machine reward 1.0

- prompt: mia has 6 cars . she drops 3 cars . how many cars does she have now ?
- completion: the answer is 3 .
- machine rationale: final answer 3 matches reference 3
- evidence: `tinyarith_r20_v1/dev/sub-6-3`

## a0011  `exact_match`  machine reward 1.0

- prompt: one day sam has 18 rocks . he gives away 0 of them . how many rocks are left ?
- completion: the answer is 18 .
- machine rationale: final answer 18 matches reference 18
- evidence: `tinyarith_r20_v1/dev/sub-18-0`
