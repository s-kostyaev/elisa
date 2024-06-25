;;; elisa.el --- Emacs Lisp Information System Assistant -*- lexical-binding: t -*-

;; Copyright (C) 2024  Free Software Foundation, Inc.

;; Author: Sergey Kostyaev <sskostyaev@gmail.com>
;; URL: http://github.com/s-kostyaev/elisa
;; Keywords: help local tools
;; Package-Requires: ((emacs "29.2") (ellama "0.9.10") (llm "0.9.1") (async "1.9.8") (plz "0.9"))
;; Version: 0.1.4
;; SPDX-License-Identifier: GPL-3.0-or-later
;; Created: 18th Feb 2024

;; This file is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 3, or (at your option)
;; any later version.

;; This file is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

;;; Commentary:
;;
;; ELISA (Emacs Lisp Information System Assistant) is a project
;; designed to help Emacs users quickly find answers to their
;; questions related to Emacs and Emacs Lisp.  Utilizing the powerful
;; Ellama package, ELISA provides accurate and relevant responses to
;; user queries, enhancing productivity and efficiency in the Emacs
;; environment.  By integrating links to the Emacs info manual after
;; answering a question, ELISA ensures that users have easy access to
;; additional information on the topic, making it an essential tool
;; for both beginners and advanced Emacs users.
;;
;; ELISA creates index from info manuals.  When you send message to
;; `elisa-chat' it search to semantically similar info nodes in index,
;; get first `elisa-limit' nodes, add it to context and send your
;; message to llm.  LLM generates answer to your message based on
;; provided context.  You can read not only answer generated by llm,
;; but also info manuals by provided links.
;;

;;; Code:
(require 'ellama)
(require 'llm)
(require 'info)
(require 'async)
(require 'dom)
(require 'shr)
(require 'plz)
(require 'json)

(defcustom elisa-embeddings-provider (progn (require 'llm-ollama)
					    (make-llm-ollama
					     :embedding-model "nomic-embed-text"))
  "Embeddings provider to generate embeddings."
  :group 'tools
  :type '(sexp :validate 'cl-struct-p))

(defcustom elisa-chat-provider (progn (require 'llm-ollama)
				      (make-llm-ollama
				       :chat-model "sskostyaev/openchat:8k-rag"
				       :embedding-model "nomic-embed-text"))
  "Chat provider."
  :group 'tools
  :type '(sexp :validate 'cl-struct-p))

(defcustom elisa-db-directory (file-truename
			       (file-name-concat
				user-emacs-directory "elisa"))
  "Directory for elisa database."
  :group 'tools
  :type 'directory)

(defcustom elisa-limit 5
  "Count quotes to pass into llm context for answer."
  :group 'tools
  :type 'integer)

(defcustom elisa-find-executable (executable-find "find")
  "Path to find executable."
  :group 'tools
  :type 'string)

(defcustom elisa-tar-executable (executable-find "tar")
  "Path to tar executable."
  :group 'tools
  :type 'string)

(defcustom elisa-sqlite-vss-version "v0.1.2"
  "Sqlite VSS version."
  :group 'tools
  :type 'string)

(defcustom elisa-semantic-split-function 'elisa-split-by-paragraph
  "Function for semantic text split."
  :group 'tools
  :type 'function)

(defcustom elisa-prompt-rewriting-enabled t
  "Enable prompt rewriting for better retrieving."
  :group 'tools
  :type 'boolean)

(defcustom elisa-rewrite-prompt-template
  "You are professional search agent. With given context and user
prompt you need to create new prompt for search. It should be
concise and useful without additional context. Response with
prompt only. User prompt:
%s"
  "Prompt template for prompt rewriting."
  :group 'tools
  :type 'string)

(defcustom elisa-searxng-url "http://localhost:8080/"
  "Searxng url for web search.  Json format should be enabled for this instance."
  :group 'tools
  :type 'string)

(defcustom elisa-pandoc-executable "pandoc"
  "Path to pandoc executable."
  :group 'tools
  :type 'string)

(defcustom elisa-webpage-extraction-function 'elisa-get-webpage-buffer
  "Function to get buffer with webpage content."
  :group 'tools
  :type 'function)

(defcustom elisa-web-search-function 'elisa-search-duckduckgo
  "Function to search the web.
Function should get prompt and return list of urls."
  :group 'tools
  :type 'function)

(defcustom elisa-web-pages-limit 10
  "Limit of web pages to parse during web search."
  :group 'tools
  :type 'integer)

(defcustom elisa-breakpoint-threshold-amount 0.4
  "Breakpoint threshold amount.
Increase it if you need decrease semantic split granularity."
  :group 'tools
  :type 'float)

(defcustom elisa-reranker-enabled t
  "Enable reranker to improve retrieving quality."
  :group 'tools
  :type 'boolean)

(defcustom elisa-reranker-url "http://127.0.0.1:8787/"
  "Reranker service url."
  :group 'tools
  :type 'string)

(defcustom elisa-reranker-similarity-threshold 0
  "Reranker similarity threshold.
If set, all quotes with similarity less than threshold will be filtered out."
  :group 'tools
  :type 'string)

(defcustom elisa-reranker-limit 20
  "Number of quotes for send to reranker."
  :group 'tools
  :type 'integer)

(defun elisa-sqlite-vss-download-url ()
  "Generate sqlite vss download url based on current system."
  (cond  ((string-equal system-type "darwin")
	  (if (string-prefix-p "aarch64" system-configuration)
	      (format
	       "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-macos-aarch64.tar.gz"
	       elisa-sqlite-vss-version
	       elisa-sqlite-vss-version)
	    (format
	     "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-macos-x86_64.tar.gz"
	     elisa-sqlite-vss-version
	     elisa-sqlite-vss-version)))
	 ((string-equal system-type "gnu/linux")
	  (format
	   "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-linux-x86_64.tar.gz"
	   elisa-sqlite-vss-version
	   elisa-sqlite-vss-version))
	 (t (user-error "Can't determine download url"))))

(defun elisa--vss-path ()
  "Path to vss sqlite extension."
  (let* ((ext (if (string-equal system-type "darwin")
		  "dylib"
		"so"))
	 (file (format "vss0.%s" ext)))
    (file-name-concat elisa-db-directory file)))

(defun elisa--vector-path ()
  "Path to vector sqlite extension."
  (let* ((ext (if (string-equal system-type "darwin")
		  "dylib"
		"so"))
	 (file (format "vector0.%s" ext)))
    (file-name-concat elisa-db-directory file)))

;;;###autoload
(defun elisa-download-sqlite-vss ()
  "Download sqlite vss."
  (interactive)
  (let ((file-name
	 (file-truename
	  (file-name-concat
	   elisa-db-directory
	   (format "sqlite-vss-%s.tar.gz" elisa-sqlite-vss-version))))
	(default-directory elisa-db-directory))
    (make-directory elisa-db-directory t)
    (url-copy-file (elisa-sqlite-vss-download-url) file-name t)
    (process-lines elisa-tar-executable "-xf" file-name)
    (delete-file file-name))
  (elisa--reopen-db))

(defun elisa-get-embedding-size ()
  "Get embedding size."
  (length (llm-embedding elisa-embeddings-provider "test")))

(defun elisa-embeddings-create-table-sql ()
  "Generate sql for create embeddings table."
  "drop table if exists elisa_embeddings;")

(defun elisa-data-embeddings-create-table-sql ()
  "Generate sql for create data embeddings table."
  (format "create virtual table if not exists data_embeddings using vss0(embedding(%d));"
	  (elisa-get-embedding-size)))

(defun elisa-data-fts-create-table-sql ()
  "Generate sql for create full text search table."
  "create virtual table if not exists data_fts using fts5(data);")

(defun elisa-info-create-table-sql ()
  "Generate sql for create info table."
  "drop table if exists info;")

(defun elisa-collections-create-table-sql ()
  "Generate sql for create collections table."
  "create table if not exists collections (name text unique);")

(defun elisa-kinds-create-table-sql ()
  "Generate sql for create kinds table."
  "create table if not exists kinds (name text unique);")

(defun elisa-fill-kinds-sql ()
  "Generate sql for fill kinds table."
  "insert into kinds (name) values ('web'), ('file'), ('info') on conflict do nothing;")

(defun elisa-data-create-table-sql ()
  "Generate sql for create data table."
  "create table if not exists data (
kind_id INTEGER,
collection_id INTEGER,
path text,
hash text,
data text,
FOREIGN KEY(kind_id) REFERENCES kinds(rowid),
FOREIGN KEY(collection_id) REFERENCES collections(rowid)
);")

(defun elisa--init-db (db)
  "Initialize elisa DB."
  (if (not (file-exists-p (elisa--vss-path)))
      (warn "Please run M-x `elisa-download-sqlite-vss' to use this package")
    (sqlite-pragma db "PRAGMA journal_mode=WAL;")
    (sqlite-load-extension
     db
     (elisa--vector-path))
    (sqlite-load-extension
     db
     (elisa--vss-path))
    (sqlite-execute db (elisa-embeddings-create-table-sql))
    (sqlite-execute db (elisa-info-create-table-sql))
    (sqlite-execute db (elisa-collections-create-table-sql))
    (sqlite-execute db (elisa-kinds-create-table-sql))
    (sqlite-execute db (elisa-fill-kinds-sql))
    (sqlite-execute db (elisa-data-create-table-sql))
    (sqlite-execute db (elisa-data-embeddings-create-table-sql))
    (sqlite-execute db (elisa-data-fts-create-table-sql))))

(defvar elisa-db (progn
		   (make-directory elisa-db-directory t)
		   (let ((db (sqlite-open (file-name-concat elisa-db-directory "elisa.sqlite"))))
		     (elisa--init-db db)
		     db)))

(defun elisa-vector-to-sqlite (data)
  "Convert DATA to sqlite vector representation."
  (format "vector_from_json(json('%s'))"
	  (json-encode data)))

(defun elisa-sqlite-escape (s)
  "Escape single quotes in S for sqlite."
  (thread-last
    s
    (string-replace "'" "''")
    (string-replace "\\" "\\\\")
    (string-replace "\0" "\n")))

(defun elisa-sqlite-format-int-list (ids)
  "Convert list of integer IDS list to sqlite list representation."
  (format
   "(%s)"
   (string-join (mapcar (lambda (id) (format "%d" id)) ids) ", ")))

(defun elisa-sqlite-format-string-list (names)
  "Convert list of string NAMES list to sqlite list representation."
  (format
   "(%s)"
   (string-join (mapcar (lambda (name)
			  (format "'%s'"
				  (elisa-sqlite-escape name))) names) ", ")))

(defun elisa-avg (lst)
  "Calculate arithmetic average value of LST."
  (let ((len (length lst))
	(sum (cl-reduce #'+ lst :initial-value 0.0)))
    (/ sum len)))

(defun elisa-std-dev (lst)
  "Calculate standart deviation value of LST."
  (let ((avg (elisa-avg lst))
	(len (length lst)))
    (sqrt (/ (cl-reduce
	      #'+
	      (mapcar
	       (lambda (x) (expt (- x avg) 2))
	       lst))
	     len))))

(defun elisa-calculate-threshold (k distances)
  "Calculate breakpoint threshold for DISTANCES based on K standard deviations."
  (+ (elisa-avg distances) (* k (elisa-std-dev distances))))

(defun elisa-parse-info-manual (name collection-name)
  "Parse info manual with NAME and save index to COLLECTION-NAME."
  (with-temp-buffer
    (info name (current-buffer))
    (let ((collection-id (or (caar (sqlite-select
				    elisa-db
				    (format
				     "select rowid from collections where name = '%s';"
				     collection-name)))
			     (progn
			       (sqlite-execute
				elisa-db
				(format
				 "insert into collections (name) values ('%s');"
				 collection-name))
			       (caar (sqlite-select
				      elisa-db
				      (format
				       "select rowid from collections where name = '%s';"
				       collection-name))))))
	  (kind-id (caar (sqlite-select
			  elisa-db "select rowid from kinds where name = 'info';")))
	  (continue t))
      (while continue
	(let* ((node-name (concat "(" (file-name-sans-extension
				       (file-name-nondirectory Info-current-file))
				  ") "
				  Info-current-node))
	       (chunks (elisa-split-semantically)))
	  (mapc
	   (lambda (text)
	     (let* ((hash (secure-hash 'sha256 text))
		    (embedding (llm-embedding elisa-embeddings-provider text))
		    (rowid
		     (if-let ((rowid (caar (sqlite-select
					    elisa-db
					    (format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';"
						    kind-id collection-id
						    (elisa-sqlite-escape node-name) hash)))))
			 nil
		       (sqlite-execute
			elisa-db
			(format
			 "insert into data(kind_id, collection_id, path, hash, data) values (%s, %s, '%s', '%s', '%s');"
			 kind-id collection-id
			 (elisa-sqlite-escape node-name) hash (elisa-sqlite-escape text)))
		       (caar (sqlite-select
			      elisa-db
			      (format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';"
				      kind-id collection-id
				      (elisa-sqlite-escape node-name) hash))))))
	       (when rowid
		 (sqlite-execute
		  elisa-db
		  (format "insert into data_embeddings(rowid, embedding) values (%s, %s);"
			  rowid (elisa-vector-to-sqlite embedding)))
		 (sqlite-execute
		  elisa-db
		  (format "insert into data_fts(rowid, data) values (%s, '%s');"
			  rowid (elisa-sqlite-escape text))))))
	   chunks)
	  (condition-case nil
	      (funcall-interactively #'Info-forward-node)
	    (error
	     (setq continue nil))))))))

(defun elisa--find-similar (text collections)
  "Find similar to TEXT results in COLLECTIONS.
Return sqlite query. For asyncronous execution."
  (let* ((rowids (flatten-tree
		  (sqlite-select
		   elisa-db
		   (format "select rowid from data where collection_id in
 (
SELECT rowid FROM collections WHERE name IN %s
);"
			   (elisa-sqlite-format-string-list collections)))))
	 (query (format "WITH
vector_search AS (
  SELECT rowid, distance
  FROM data_embeddings
  WHERE vss_search(embedding, %s)
  ORDER BY distance ASC
  LIMIT 40
),
semantic_search AS (
  SELECT rowid, RANK () OVER (ORDER BY distance ASC) AS rank
  FROM vector_search
  WHERE rowid IN %s
  ORDER BY distance ASC
  LIMIT 20
),
keyword_search AS (
  SELECT rowid, RANK () OVER (ORDER BY bm25(data_fts) ASC) AS rank
  FROM data_fts
  WHERE rowid in %s and data_fts MATCH '%s'
  ORDER BY bm25(data_fts) ASC
  LIMIT 20
),
hybrid_search AS (
SELECT
  COALESCE(semantic_search.rowid, keyword_search.rowid) AS rowid,
  COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
  COALESCE(1.0 / (60 + keyword_search.rank), 0.0) AS score
FROM semantic_search
FULL OUTER JOIN keyword_search ON semantic_search.rowid = keyword_search.rowid
ORDER BY score DESC
LIMIT %d
)
SELECT
  hybrid_search.rowid
FROM hybrid_search
;
"
			(elisa-vector-to-sqlite
			 (llm-embedding elisa-embeddings-provider text))
			(elisa-sqlite-format-int-list rowids)
			(elisa-sqlite-format-int-list rowids)
			(elisa-fts-query text)
			(elisa-get-limit))))
    query))

(defun elisa-find-similar (text collections on-done)
  "Find similar to TEXT results in COLLECTIONS.
Evaluate ON-DONE with result."
  (message "searching in collected data")
  (elisa--async-do
   (lambda () (elisa--find-similar text collections))
   on-done))

(defun elisa--split-by (func)
  "Split buffer content to list by FUNC."
  (let ((pt (point-min))
	(result nil))
    (save-excursion
      (goto-char (point-min))
      (while (< (point) (point-max))
	(funcall func)
	(push (buffer-substring-no-properties pt (point)) result)
	(setq pt (point)))
      (nreverse (cl-remove-if #'string-empty-p result)))))

(defun elisa-split-by-sentence ()
  "Split byffer to list of sentences."
  (elisa--split-by #'forward-sentence))

(defun elisa-split-by-paragraph ()
  "Split buffer to list of paragraphs."
  (elisa--split-by #'forward-paragraph))

(defun elisa-dot-product (v1 v2)
  "Calculate the dot produce of vectors V1 and V2."
  (let ((result 0))
    (dotimes (i (length v1))
      (setq result (+ result (* (aref v1 i) (aref v2 i)))))
    result))

(defun elisa-magnitude (v)
  "Calculate magnitude of vector V."
  (let ((sum 0))
    (dotimes (i (length v))
      (setq sum (+ sum (* (aref v i) (aref v i)))))
    (sqrt sum)))

(defun elisa-cosine-similarity (v1 v2)
  "Calculate the cosine similarity of V1 and V2.
The return is a floating point number between 0 and 1, where the
closer it is to 1, the more similar it is."
  (let ((dot-product (elisa-dot-product v1 v2))
        (v1-magnitude (elisa-magnitude v1))
        (v2-magnitude (elisa-magnitude v2)))
    (if (and v1-magnitude v2-magnitude)
        (/ dot-product (* v1-magnitude v2-magnitude))
      0)))

(defun elisa-cosine-distance (v1 v2)
  "Calculate cosine-distance between V1 and V2."
  (- 1 (elisa-cosine-similarity v1 v2)))

(defun elisa--similarities (list)
  "Calculate cosine similarities between neighbour elements in LIST."
  (let ((head (car list))
	(tail (cdr list))
	(result nil))
    (while tail
      (push (elisa-cosine-similarity head (car tail)) result)
      (setq head (car tail))
      (setq tail (cdr tail)))
    (nreverse result)))

(defun elisa--distances (list)
  "Calculate cosine distances between neighbour elements in LIST."
  (let ((head (car list))
	(tail (cdr list))
	(result nil))
    (while tail
      (push (elisa-cosine-distance head (car tail)) result)
      (setq head (car tail))
      (setq tail (cdr tail)))
    (nreverse result)))

(defun elisa-split-semantically (&rest args)
  "Split buffer data semantically.
ARGS contains keys for fine control.

:function FUNC -- FUNC is a function for split buffer into chunks.

:threshold-amount K -- K is a breakpoint threshold amount.

than T, it will be packed into single semantic chunk."
  (when-let* ((func (or (plist-get args :function) elisa-semantic-split-function))
	      (k (or (plist-get args :threshold-amount) elisa-breakpoint-threshold-amount))
	      (chunks (funcall func))
	      (embeddings (cl-remove-if
			   #'not
			   (mapcar (lambda (s)
				     (when (length> (string-trim s) 0)
				       (llm-embedding elisa-embeddings-provider s)))
				   chunks)))
	      (distances (elisa--distances embeddings))
	      (threshold (elisa-calculate-threshold k distances))
	      (current (car chunks))
	      (tail (cdr chunks)))
    (let* ((result nil))
      (mapc
       (lambda (el)
	 (if (<= el threshold)
	     (setq current (concat current (car tail)))
	   (push current result)
	   (setq current (car tail)))
	 (setq tail (cdr tail)))
       distances)
      (push current result)
      (cl-remove-if
       #'string-empty-p
       (mapcar (lambda (s)
		 (if s
		     (string-trim s)
		   ""))
	       (nreverse result))))))

(defun elisa-search-duckduckgo (prompt)
  "Search duckduckgo for PROMPT and return list of urls."
  (let* ((url (format "https://duckduckgo.com/html/?q=%s" (url-hexify-string prompt)))
	 (buffer-name (plz 'get url :as 'buffer
			:headers `(("Accept" . ,eww-accept-content-types)
				   ("Accept-Encoding" . "gzip")
				   ("User-Agent" . ,(url-http--user-agent-default-string))))))
    (with-current-buffer buffer-name
      (goto-char (point-min))
      (search-forward "<!DOCTYPE")
      (beginning-of-line)
      (cl-remove-if
       #'string-empty-p
       (cl-remove-duplicates
	(mapcar
	 (lambda (el)
	   (when el
	     (string-trim-right
	      (url-unhex-string
	       (cdar (url-parse-args (or (dom-attr el 'href) ""))))
	      "[&\\?].*")))
	 (dom-by-tag
	  (libxml-parse-html-region
	   (point) (point-max))
	  'a))
	:test 'string-equal)))))

(defun elisa-search-searxng (prompt)
  "Search searxng for PROMPT and return list of urls.
You can customize `elisa-searxng-url' to use non local instance."
  (let ((url (format "%s/search?format=json&q=%s" elisa-searxng-url (url-hexify-string prompt))))
    (thread-last
      (plz 'get url :as 'json-read)
      (alist-get 'results)
      (mapcar (lambda (el) (alist-get 'url el))))))

(defun elisa-get-webpage-buffer (url)
  "Get buffer with URL content."
  (let ((buffer-name (ignore-errors
		       (plz 'get url :as 'buffer
			 :headers `(("Accept" . ,eww-accept-content-types)
				    ("Accept-Encoding" . "gzip")
				    ("User-Agent" . ,(url-http--user-agent-default-string))))))
	;; fix one word lines for async execution
	(shr-use-fonts nil)
	(shr-width (- ellama-long-lines-length 5)))
    (when buffer-name
      (with-current-buffer buffer-name
	(goto-char (point-min))
	(or (search-forward "<!DOCTYPE" nil t)
            (search-forward "<html" nil t))
	(beginning-of-line)
	(kill-region (point-min) (point))
	(ignore-errors
	  (shr-insert-document (libxml-parse-html-region (point-min) (point-max))))
	(goto-char (point-min))
	(or (search-forward "<!DOCTYPE" nil t)
            (search-forward "<html" nil t))
	(beginning-of-line)
	(kill-region (point) (point-max))
	buffer-name))))

(defun elisa-get-webpage-buffer-pandoc (url)
  "Get buffer with URL content translated to markdown with pandoc."
  (let ((buffer-name (plz 'get url :as 'buffer)))
    (with-current-buffer buffer-name
      (shell-command-on-region
       (point-min) (point-max)
       (format "%s -f html --to plain"
	       (executable-find elisa-pandoc-executable))
       buffer-name t)
      buffer-name)))

(defun elisa-fts-query (prompt)
  "Return fts match query for PROMPT."
  (thread-last
    prompt
    (string-trim)
    (downcase)
    (string-replace "-" " ")
    (replace-regexp-in-string "[^[:alnum:] ]+" "")
    (string-trim)
    (replace-regexp-in-string "[[:space:]]+" " OR ")))

(defun elisa--rerank-request (prompt ids)
  "Generate rerank request body for PROMPT and IDS."
  (let ((docs
	 (mapcar
	  (lambda (row)
	    (let ((id (cl-first row))
		  (text (cl-second row)))
	      `(("id" . ,id) ("text" . ,text))))
	  (sqlite-select
	   elisa-db
	   (format
	    "select rowid, data from data where rowid in %s;"
	    (elisa-sqlite-format-int-list ids))))))
    (json-encode `(("query" . ,prompt)
		   ("documents" . ,docs)))))

(defun elisa--do-rerank-request (prompt ids)
  "Call rerank service for PROMPT and IDS."
  (seq--into-list
   (alist-get 'data
	      (plz 'post (format "%s/api/v1/rerank"
				 (string-remove-suffix "/" elisa-reranker-url))
		:headers `(("Content-Type" . "application/json"))
		:body-type 'text
		:body (elisa--rerank-request prompt ids)
		:as #'json-read))))

(defun elisa-rerank (prompt ids)
  "Rerank IDS according to PROMPT and return top `elisa-limit' IDS."
  (let ((data (elisa--do-rerank-request prompt ids)))
    (mapcar (lambda (elt)
	      (alist-get 'id elt))
	    (take elisa-limit
		  (if elisa-reranker-similarity-threshold
		      (cl-remove-if (lambda (obj)
				      (< (alist-get 'similarity obj)
					 elisa-reranker-similarity-threshold))
				    data)
		    data)))))

(defun elisa-get-limit ()
  "Limit for elisa hybrid search."
  (if elisa-reranker-enabled
      elisa-reranker-limit
    elisa-limit))

(defun elisa--web-search (prompt)
  "Search the web for PROMPT.
Return sqlite query that extract data for adding to context."
  (sqlite-execute
   elisa-db
   (format
    "insert into collections (name) values ('%s') on conflict do nothing;"
    (elisa-sqlite-escape prompt)))
  (let* ((kind-id (caar (sqlite-select
			 elisa-db "select rowid from kinds where name = 'web';")))
	 (collection-id (caar (sqlite-select
			       elisa-db
			       (format
				"select rowid from collections where name = '%s';"
				(elisa-sqlite-escape prompt)))))
	 (urls (funcall elisa-web-search-function prompt))
	 (collected-pages 0))
    (mapc (lambda (url)
	    (when (<= collected-pages elisa-web-pages-limit)
	      (message "collecting data from %s" url)
	      (mapc
	       (lambda (chunk)
		 (let* ((hash (secure-hash 'sha256 chunk))
			(embedding (llm-embedding elisa-embeddings-provider chunk))
			(rowid
			 (if-let ((rowid (caar (sqlite-select
						elisa-db
						(format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';" kind-id collection-id url hash)))))
			     nil
			   (sqlite-execute
			    elisa-db
			    (format
			     "insert into data(kind_id, collection_id, path, hash, data) values (%s, %s, '%s', '%s', '%s');"
			     kind-id collection-id url hash (elisa-sqlite-escape chunk)))
			   (caar (sqlite-select
				  elisa-db
				  (format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';" kind-id collection-id url hash))))))
		   (when rowid
		     (sqlite-execute
		      elisa-db
		      (format "insert into data_embeddings(rowid, embedding) values (%s, %s);"
			      rowid (elisa-vector-to-sqlite embedding)))
		     (sqlite-execute
		      elisa-db
		      (format "insert into data_fts(rowid, data) values (%s, '%s');"
			      rowid (elisa-sqlite-escape chunk))))))
	       (elisa-extact-webpage-chunks url))
	      (cl-incf collected-pages)))
	  urls)))

;;;###autoload
(defun elisa-web-search (prompt)
  "Search the web for PROMPT."
  (interactive "sAsk elisa with web search: ")
  (message "searching the web")
  (elisa--async-do
   (lambda () (elisa--web-search prompt))
   (lambda (_)
     (elisa-find-similar
      prompt (list prompt)
      (lambda (query) (elisa-retrieve-ask query prompt))))))

(defun elisa-retrieve-ask (query prompt)
  "Retrieve data with QUERY and ask elisa for PROMPT."
  (elisa--async-do
   (lambda () (let* ((raw-ids (flatten-tree (sqlite-select elisa-db query)))
		     (ids (if elisa-reranker-enabled
			      (elisa-rerank prompt raw-ids)
			    (take elisa-limit raw-ids))))
		(sqlite-select
		 elisa-db
		 (format
		  "SELECT k.name, d.path, d.data
FROM data AS d
LEFT JOIN kinds k ON k.rowid = d.kind_id
WHERE d.rowid in %s;"
		  (elisa-sqlite-format-int-list ids)))))
   (lambda (result)
     (mapc
      (lambda (row)
	(when-let ((kind (cl-first row))
		   (path (cl-second row))
		   (text (cl-third row)))
	  (pcase kind
	    ("web"
	     (ellama-context-add-webpage-quote-noninteractive path path text))
	    ("file"
	     (ellama-context-add-file-quote-noninteractive path text))
	    ("info"
	     (ellama-context-add-info-node-quote-noninteractive path text)))))
      result)
     (ellama-chat prompt nil :provider elisa-chat-provider))))

(defun elisa-get-builtin-manuals ()
  "Get builtin manual names list."
  (mapcar
   #'file-name-base
   (cl-remove-if-not
    (lambda (s) (or (string-suffix-p ".info" s)
		    (string-suffix-p ".info.gz" s)))
    (directory-files (with-temp-buffer
		       (info "emacs" (current-buffer))
		       (file-name-directory Info-current-file))))))

(defun elisa-get-external-manuals ()
  "Get external manual names list."
  (seq-uniq
   (mapcar
    #'file-name-base
    (process-lines
     elisa-find-executable
     (file-truename
      (file-name-concat user-emacs-directory "elpa"))
     "-name" "*.info"))))

;;;###autoload
(defun elisa-parse-builtin-manuals ()
  "Parse builtin manuals."
  (interactive)
  (mapc (lambda (s)
	  (ignore-errors (elisa-parse-info-manual s "builtin manuals")))
	(elisa-get-builtin-manuals)))

;;;###autoload
(defun elisa-parse-external-manuals ()
  "Parse external manuals."
  (interactive)
  (mapc (lambda (s)
	  (ignore-errors (elisa-parse-info-manual s "external manuals")))
	(elisa-get-external-manuals)))

;;;###autoload
(defun elisa-parse-all-manuals ()
  "Parse all manuals."
  (interactive)
  (elisa-parse-builtin-manuals)
  (elisa-parse-external-manuals))

(defun elisa--reopen-db ()
  "Reopen database."
  (let ((db (sqlite-open (file-name-concat elisa-db-directory "elisa.sqlite"))))
    (elisa--init-db db)
    (setq elisa-db db)))

(defun elisa--async-do (func &optional on-done)
  "Do FUNC asyncronously.
Call ON-DONE callback with result as an argument after FUNC evaluation done."
  (async-start `(lambda ()
		  ,(async-inject-variables "elisa-embeddings-provider")
		  ,(async-inject-variables "elisa-db-directory")
		  ,(async-inject-variables "elisa-find-executable")
		  ,(async-inject-variables "elisa-tar-executable")
		  ,(async-inject-variables "elisa-prompt-rewriting-enabled")
		  ,(async-inject-variables "elisa-rewrite-prompt-template")
		  ,(async-inject-variables "elisa-semantic-split-function")
		  ,(async-inject-variables "elisa-webpage-extraction-function")
		  ,(async-inject-variables "elisa-web-search-function")
		  ,(async-inject-variables "elisa-searxng-url")
		  ,(async-inject-variables "elisa-web-pages-limit")
		  ,(async-inject-variables "elisa-breakpoint-threshold-amount")
		  ,(async-inject-variables "elisa-pandoc-executable")
		  ,(async-inject-variables "ellama-long-lines-length")
		  ,(async-inject-variables "load-path")
		  (require 'elisa)
		  (,func))
	       (lambda (res)
		 (sqlite-close elisa-db)
		 (elisa--reopen-db)
		 (when on-done
		   (funcall on-done res))
		 (message "%s done."
			  func))))

(defun elisa-extact-webpage-chunks (url)
  "Extract semantic chunks for webpage fetched from URL."
  (when-let ((buf (funcall elisa-webpage-extraction-function url)))
    (with-current-buffer buf
      (elisa-split-semantically))))

;;;###autoload
(defun elisa-async-parse-builtin-manuals ()
  "Parse builtin manuals asyncronously."
  (interactive)
  (message "Begin parsing builtin manuals.")
  (elisa--async-do 'elisa-parse-builtin-manuals))

;;;###autoload
(defun elisa-async-parse-external-manuals ()
  "Parse external manuals asyncronously."
  (interactive)
  (message "Begin parsing external manuals.")
  (elisa--async-do 'elisa-parse-external-manuals))

;;;###autoload
(defun elisa-async-parse-all-manuals ()
  "Parse all manuals asyncronously."
  (interactive)
  (message "Begin parsing manuals.")
  (elisa--async-do 'elisa-parse-all-manuals))

;;;###autoload
(defun elisa-chat (prompt &optional collections)
  "Send PROMPT to elisa.
Find similar quotes in COLLECTIONS and add it to context."
  (interactive "sAsk elisa: ")
  (let ((cols (or collections '("builtin manuals" "external manuals"))))
    (elisa-find-similar
     prompt cols
     (lambda (query) (elisa-retrieve-ask query prompt)))))

(provide 'elisa)
;;; elisa.el ends here.
