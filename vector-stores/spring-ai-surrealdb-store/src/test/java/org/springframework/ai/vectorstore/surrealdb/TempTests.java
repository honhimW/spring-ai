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

import com.surrealdb.Response;
import com.surrealdb.Surreal;
import com.surrealdb.Value;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

/**
 * @author honhimW
 * @since 2025-04-10
 */

public class TempTests {

	private static Surreal surreal;

	@BeforeAll
	static void beforeAll() {
		surreal = new Surreal();
		surreal.connect("memory");
		surreal.useNs("spring-ai");
		surreal.useDb("test");
	}

	@AfterAll
	static void afterAll() {
		surreal.close();
	}

	@Test
	void defineIndex() throws Exception {
		Response response = surreal.query("""
				DEFINE INDEX IF NOT EXISTS %s
				ON %s
				FIELDS %s
				%s
				DIMENSION %d
				DIST %s
				TYPE %s;
				"""
				.formatted(
						"spring_ai_index",
						"embedding",
						"embedding",
						"HNSW",
						1024,
						"COSINE",
						"F32"
				));
		Value take = response.take(0);
		System.out.println(take);
	}

	@Test
	void vectorSearch() throws Exception {
		Map<String, Object> bindings = new HashMap<>();
//		float[] person_embedding = {0.15f, 0.25f, 0.35f, 0.45f};
//		bindings.put("person_embedding", person_embedding);
		Response response = surreal.queryBind("""
				-- Create a dataset of actors with embeddings and flags
				CREATE actor:1 SET name = 'Actor 1', embedding = [0.1, 0.2, 0.3, 0.4], flag = true;
				CREATE actor:2 SET name = 'Actor 2', embedding = [0.2, 0.1, 0.4, 0.3], flag = false;
				CREATE actor:3 SET name = 'Actor 3', embedding = [0.4, 0.3, 0.2, 0.1], flag = true;
				CREATE actor:4 SET name = 'Actor 4', embedding = [0.3, 0.4, 0.1, 0.2], flag = true;
				
				-- Define an HNSW index on the actor table
				DEFINE INDEX hnsw_pts ON actor FIELDS embedding HNSW DIMENSION 4;
				
				-- Select actors who look like you and have won an Oscar
				SELECT id, flag, vector::distance::knn() AS distance FROM actor WHERE flag = true AND embedding <|2,40|> $person_embedding ORDER BY distance;
				""", bindings);
		Value take = response.take(0);
		System.out.println(take);
	}

}
