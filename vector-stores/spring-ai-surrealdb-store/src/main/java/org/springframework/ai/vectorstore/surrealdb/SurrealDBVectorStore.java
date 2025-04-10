/*
 * Copyright 2025 the original honhimW.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.vectorstore.surrealdb;

import com.surrealdb.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptionsBuilder;
import org.springframework.ai.observation.conventions.VectorStoreProvider;
import org.springframework.ai.observation.conventions.VectorStoreSimilarityMetric;
import org.springframework.ai.vectorstore.AbstractVectorStoreBuilder;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.observation.AbstractObservationVectorStore;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationContext;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.lang.Nullable;
import org.springframework.util.StringUtils;

import java.lang.Object;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * @author honhimW
 * @since 1.0.0
 */

public class SurrealDBVectorStore extends AbstractObservationVectorStore implements InitializingBean {

	private static final Logger logger = LoggerFactory.getLogger(SurrealDBVectorStore.class);

	public static final String DEFAULT_INDEX_NAME = "spring_ai_index";

	public static final String DEFAULT_CONTENT_FIELD_NAME = "content";

	public static final String DEFAULT_EMBEDDING_FIELD_NAME = "embedding";

	public static final String DEFAULT_TABLE_NAME = "spring_ai_embedding";

	public static final Algorithm DEFAULT_VECTOR_ALGORITHM = Algorithm.HNSW;

	public static final Type DEFAULT_VECTOR_TYPE = Type.F64;

	public static final Distance DEFAULT_VECTOR_DISTANCE = Distance.COSINE;

	public static final UpType DEFAULT_UPDATE_TYPE = UpType.CONTENT;

	private static final String RECORD_ID_FIELD = "id";

	private final Surreal surreal;

	private final boolean initializeSchema;

	private final String indexName;

	private final String contentFieldName;

	private final String embeddingFieldName;

	private final String tableName;

	private final Algorithm vectorAlgorithm;

	private final Type vectorType;

	private final Distance vectorDistance;

	private final UpType updateType;

	/**
	 * Creates a new AbstractObservationVectorStore instance with the specified builder
	 * settings. Initializes observation-related components and the embedding model.
	 *
	 * @param builder the builder containing configuration settings
	 */
	public SurrealDBVectorStore(Builder builder) {
		super(builder);
		this.initializeSchema = builder.initializeSchema;
		this.surreal = builder.surreal;
		this.indexName = builder.indexName;
		this.contentFieldName = builder.contentFieldName;
		this.embeddingFieldName = builder.embeddingFieldName;
		this.tableName = builder.tableName;
		this.vectorAlgorithm = builder.vectorAlgorithm;
		this.vectorType = builder.vectorType;
		this.vectorDistance = builder.vectorDistance;
		this.updateType = builder.updateType;
	}

	public static Builder builder(Surreal surreal, EmbeddingModel embeddingModel) {
		return new Builder(surreal, embeddingModel);
	}

	public Surreal getSurreal() {
		return this.surreal;
	}

	@Override
	public void doAdd(List<Document> documents) {
		List<float[]> embeddings = this.embeddingModel.embed(documents, EmbeddingOptionsBuilder.builder().build(),
				this.batchingStrategy);
		for (Document document : documents) {
			Map<String, Object> fields = new HashMap<>();
			fields.put(RECORD_ID_FIELD, document.getId());
			fields.put(this.embeddingFieldName, embeddings.get(documents.indexOf(document)));
			fields.put(this.contentFieldName, document.getText());
			fields.putAll(document.getMetadata());
			this.surreal.upsert(this.tableName, updateType, fields);
		}
	}

	@Override
	public void doDelete(List<String> idList) {
		RecordId[] recordIds = idList.stream()
				.map(this::recordId)
				.toArray(RecordId[]::new);
		this.surreal.delete(recordIds);
	}

	@Override
	public List<Document> doSimilaritySearch(SearchRequest request) {
		String query = request.getQuery();
		int topK = request.getTopK();
		double similarityThreshold = request.getSimilarityThreshold();
		Filter.Expression filterExpression = request.getFilterExpression();
		Map<String, Object> bindings = new LinkedHashMap<>();
		Response response = this.surreal.queryBind("""
				SELECT
				""", bindings);
		Value take = response.take(0);

		return List.of();
	}

	@Override
	public VectorStoreObservationContext.Builder createObservationContextBuilder(String operationName) {
		VectorStoreObservationContext.Builder builder = VectorStoreObservationContext.builder(VectorStoreProvider.SURREALDB.value(), operationName)
				.collectionName(this.tableName)
				.dimensions(this.embeddingModel.dimensions())
				.fieldName(this.embeddingFieldName);
		return switch (this.vectorDistance) {
			case COSINE -> builder.similarityMetric(VectorStoreSimilarityMetric.COSINE.value());
			case EUCLIDEAN -> builder.similarityMetric(VectorStoreSimilarityMetric.EUCLIDEAN.value());
			case MANHATTAN -> builder.similarityMetric(VectorStoreSimilarityMetric.MANHATTAN.value());
			default -> builder;
		};
	}

	@Override
	public void afterPropertiesSet() throws Exception {
		if (!this.initializeSchema) {
			return;
		}

		// Using `IF NOT EXISTS` to avoid errors if the index already exists.
		this.surreal.query("""
			DEFINE INDEX IF NOT EXISTS %s
			ON %s
			FIELDS %s
			%s
			DIMENSION %d
			DIST %s
			TYPE %s;
			"""
				.formatted(
						indexName,
						tableName,
						embeddingFieldName,
						vectorAlgorithm,
						this.embeddingModel.dimensions(),
						vectorDistance,
						vectorType
				));

	}

	public enum Algorithm {

		HNSW, MTREE

	}

	public enum Type {

		F64, F32, I64, I32, I16

	}

	public enum Distance {

		COSINE, EUCLIDEAN, MANHATTAN, MINKOWSKI

	}

	public static class Builder extends AbstractVectorStoreBuilder<Builder> {

		private final Surreal surreal;

		private String indexName = DEFAULT_INDEX_NAME;

		private String tableName = DEFAULT_TABLE_NAME;

		private String contentFieldName = DEFAULT_CONTENT_FIELD_NAME;

		private String embeddingFieldName = DEFAULT_EMBEDDING_FIELD_NAME;

		private Algorithm vectorAlgorithm = DEFAULT_VECTOR_ALGORITHM;

		private Type vectorType = DEFAULT_VECTOR_TYPE;

		private Distance vectorDistance = DEFAULT_VECTOR_DISTANCE;

		private UpType updateType = DEFAULT_UPDATE_TYPE;

		private boolean initializeSchema = false;

		public Builder(Surreal surreal, EmbeddingModel embeddingModel) {
			super(embeddingModel);
			this.surreal = surreal;
		}

		public Builder indexName(String indexName) {
			if (StringUtils.hasText(indexName)) {
				this.indexName = indexName;
			}
			return this;
		}

		public Builder tableName(String tableName) {
			if (StringUtils.hasText(this.tableName)) {
				this.tableName = tableName;
			}
			return this;
		}

		public Builder contentFieldName(String fieldName) {
			if (StringUtils.hasText(fieldName)) {
				this.contentFieldName = fieldName;
			}
			return this;
		}

		public Builder embeddingFieldName(String fieldName) {
			if (StringUtils.hasText(fieldName)) {
				this.embeddingFieldName = fieldName;
			}
			return this;
		}

		public Builder vectorAlgorithm(@Nullable Algorithm algorithm) {
			if (algorithm != null) {
				this.vectorAlgorithm = algorithm;
			}
			return this;
		}

		public Builder vectorType(@Nullable Type type) {
			if (type != null) {
				this.vectorType = type;
			}
			return this;
		}

		public Builder vectorDistance(@Nullable Distance distance) {
			if (distance != null) {
				this.vectorDistance = distance;
			}
			return this;
		}

		public Builder updateType(@Nullable UpType updateType) {
			if (updateType != null) {
				this.updateType = updateType;
			}
			return this;
		}

		public Builder initializeSchema(boolean initializeSchema) {
			this.initializeSchema = initializeSchema;
			return this;
		}

		@Override
		public VectorStore build() {
			return new SurrealDBVectorStore(this);
		}
	}

	private RecordId recordId(String id) {
		return new RecordId(this.tableName, id);
	}

}
