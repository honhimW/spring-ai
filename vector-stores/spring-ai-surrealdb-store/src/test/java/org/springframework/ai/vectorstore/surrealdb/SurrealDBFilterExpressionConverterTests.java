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

import org.junit.jupiter.api.Test;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.filter.Filter;

import java.util.List;

/**
 * @author honhimW
 * @since 2025-04-11
 */

public class SurrealDBFilterExpressionConverterTests {

	static SurrealDBFilterExpressionConverter converter = new SurrealDBFilterExpressionConverter();

	@Test
	void eq() {
		Filter.Expression expression = new Filter.Expression(Filter.ExpressionType.EQ, new Filter.Key("flag"), new Filter.Value(false));
		String s = converter.convertExpression(expression);
		System.out.println(s);
	}

	@Test
	void in() {
		Filter.Expression expression = new Filter.Expression(Filter.ExpressionType.IN, new Filter.Key("flag"), new Filter.Value(List.of(true, false)));
		String s = converter.convertExpression(expression);
		System.out.println(s);
	}

}
