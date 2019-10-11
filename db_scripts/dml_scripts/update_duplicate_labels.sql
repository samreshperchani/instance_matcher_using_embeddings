USE [instance_matcher_db]
GO

UPDATE e_l
SET
revised_id =CONCAT('<http://dbkwik.webdatacommons.org/resource/', REPLACE(e_l.label, ' ','_'),'>')
FROM dbo.entity_labels  e_l
inner join dbo.entity_dup_labels ins
on ins.label = e_l.label
and ins.count > 1 


USE [instance_matcher_db]
GO
UPDATE e_l
SET
revised_id =CONCAT('<http://dbkwik.webdatacommons.org/resource/category:', REPLACE(e_l.label, ' ','_'),'>')
FROM dbo.category_labels  e_l
inner join dbo.category_dup_labels cat
on cat.label = e_l.label
and cat.count > 1 



USE [instance_matcher_db]
GO
UPDATE e_l
SET
revised_id =CONCAT('<http://dbkwik.webdatacommons.org/property/', REPLACE(e_l.label, ' ','_'),'>')
FROM dbo.prop_labels  e_l
inner join dbo.prop_dup_labels prop
on prop.label = e_l.label
and prop.count > 1 



USE [instance_matcher_db]
GO
UPDATE e_l
SET
revised_id =CONCAT('<http://dbkwik.webdatacommons.org/class/', REPLACE(e_l.label, ' ','_'),'>')
FROM dbo.class_labels  e_l
inner join dbo.class_dup_labels class
on class.label = e_l.label
and class.count > 1 