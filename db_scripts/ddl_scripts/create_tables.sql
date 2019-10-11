USE [instance_matcher_db]
GO

/****** Object:  Table [dbo].[entity_labels]    Script Date: 10/11/2019 4:01:58 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[entity_labels](
	[entity_id] [varchar](max) NULL,
	[label] [varchar](max) NULL,
	[wiki_name] [varchar](max) NULL,
	[revised_id] [varchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO

USE [instance_matcher_db]
GO

/****** Object:  Table [dbo].[category_labels]    Script Date: 10/11/2019 4:01:58 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[category_labels](
	[category_id] [varchar](max) NULL,
	[label] [varchar](max) NULL,
	[wiki_name] [varchar](max) NULL,
	[revised_id] [varchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO

USE [instance_matcher_db]
GO

/****** Object:  Table [dbo].[prop_labels]    Script Date: 10/11/2019 4:01:58 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[prop_labels](
	[prop_id] [varchar](max) NULL,
	[label] [varchar](max) NULL,
	[wiki_name] [varchar](max) NULL,
	[revised_id] [varchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO


USE [instance_matcher_db]
GO

/****** Object:  Table [dbo].[class_labels]    Script Date: 10/11/2019 4:01:58 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[class_labels](
	[class_id] [varchar](max) NULL,
	[label] [varchar](max) NULL,
	[wiki_name] [varchar](max) NULL,
	[revised_id] [varchar](max) NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO



USE [instance_matcher_db]
GO

/****** Object:  Table [dbo].[inst_dup_labels]    Script Date: 10/11/2019 4:06:04 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[entity_dup_labels](
	[label] [nvarchar](4000) NULL,
	[count] [int] NULL
) ON [PRIMARY]
GO


USE [instance_matcher_db]
GO
/****** Object:  Table [dbo].[inst_dup_labels]    Script Date: 10/11/2019 4:06:04 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[category_dup_labels](
	[label] [nvarchar](4000) NULL,
	[count] [int] NULL
) ON [PRIMARY]
GO



USE [instance_matcher_db]
GO
/****** Object:  Table [dbo].[prop_dup_labels]    Script Date: 10/11/2019 4:06:04 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[prop_dup_labels](
	[label] [nvarchar](4000) NULL,
	[count] [int] NULL
) ON [PRIMARY]
GO


USE [instance_matcher_db]
GO
/****** Object:  Table [dbo].[class_matcher_db]******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[class_dup_labels](
	[label] [nvarchar](4000) NULL,
	[count] [int] NULL
) ON [PRIMARY]
GO