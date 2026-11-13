# Withdrawn development-role contract correction

The accompanying `withdrawn_hey_broad_development_role_correction_v1.patch` is retained as an abandoned proposal, **not for application**. Its mechanical applicability to GPU `9fd2d76` was tested, but that did not establish that it respected the GPU session's user instructions.

Mac's active objective and earlier coordination allowed explicit reuse of development with advance rules. Mac therefore proposed removing a prospective contract sentence that required unconsumed development. GPU comment `6866411177` subsequently clarified that its own current direct user objective explicitly says “使用未消费开发数据选阈值，比较冻结v30与候选”. This evidence changes the decision: preserve that requirement and withdraw the proposed deletion. The shared route can satisfy both sessions by using the new locked development for selection and retaining old development only as labelled regression in that experiment. No historical result, user goal or active training is changed.

The generic raw collector can record either `new-development` or `reused-development` without choosing thresholds; which role may select a threshold remains governed by the active experiment contract. It does not override the clarified GPU requirement or independent acceptance.
